"""
Chunked aggregations and statistics for Exorde stratified sample CSV.
Used by analysis_events.ipynb — keeps the notebook readable.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHUNK_SIZE = 50_000
MAX_SENTIMENT_PER_PLATFORM = 12_000  # per event, for plots/tests
TOP_PLATFORMS_PLOT = 8
TOP_LANG_PLOT = 10

DOMAIN_RE = re.compile(r"https?://(?:www\.)?([^/]+)")

# Keyword bundles (lowercase substrings). Tune for precision/recall trade-offs.
EVENTS: dict[str, dict[str, Any]] = {
    "us_politics": {
        "label": "US politics (post–US election, Dec 2024 week)",
        "keywords": [
            "trump",
            "biden",
            "election",
            "president",
            "gop",
            "democrat",
            "republican",
            "kamala",
            "harris",
            "white house",
            "electoral",
            "congress",
        ],
    },
    "syria": {
        "label": "Syria conflict (rebel offensive, Dec 2024)",
        "keywords": [
            "syria",
            "syrian",
            "damascus",
            "aleppo",
            "homs",
            "idlib",
            "assad",
            "rebel",
            "palmyra",
            "deir",
        ],
    },
    "romania": {
        "label": "Romania (election annulment / political crisis)",
        "keywords": [
            "romania",
            "bucharest",
            "iohannis",
            "geoana",
            "bucuresti",
        ],
    },
    "sri_lanka": {
        "label": "Sri Lanka / NPP",
        "keywords": [
            "sri lanka",
            "lanka",
            "npp",
            "anura",
            "kumara",
            "dissanayake",
            "colombo",
            "jaffna",
            "parliament",
        ],
    },
    "gaza_israel": {
        "label": "Gaza / Israel",
        "keywords": [
            "gaza",
            "israel",
            "hamas",
            "palestine",
            "palestinian",
            "idf",
            "netanyahu",
            "ceasefire",
            "west bank",
            "jerusalem",
        ],
    },
}


def get_platform(url: object) -> str:
    if not isinstance(url, str) or not url:
        return "missing"
    m = DOMAIN_RE.match(url.strip())
    return m.group(1).lower() if m else "missing"


def _event_mask(blob: pd.Series, keywords: list[str]) -> pd.Series:
    mask = pd.Series(False, index=blob.index)
    for kw in keywords:
        mask = mask | blob.str.contains(re.escape(kw), case=False, na=False, regex=True)
    return mask


def run_chunked_pipeline(csv_path: str, chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)

    n_rows = 0
    sentiment_missing = 0
    url_missing = 0
    theme_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()

    # Global daily: date_str -> {n, sum_s}
    daily_global: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "sum_s": 0.0}
    )

    # Per event: date -> platform -> count; date -> lang -> count; hour -> platform (for sparse calendar coverage)
    ed: dict[str, dict[str, defaultdict[Any, Any]]] = {
        e: {
            "day_plat": defaultdict(lambda: defaultdict(int)),
            "day_lang": defaultdict(lambda: defaultdict(int)),
            "hour_plat": defaultdict(lambda: defaultdict(int)),
        }
        for e in EVENTS
    }
    # Daily volume + sentiment per event
    ed_daily: dict[str, dict[str, dict[str, float]]] = {
        e: defaultdict(lambda: {"n": 0, "sum_s": 0.0}) for e in EVENTS
    }
    # Hourly volume + sentiment (UTC), for samples with few distinct calendar days
    ed_hourly: dict[str, dict[str, dict[str, float]]] = {
        e: defaultdict(lambda: {"n": 0, "sum_s": 0.0}) for e in EVENTS
    }

    # Reservoir: event -> platform -> list of sentiment
    reservoirs: dict[str, dict[str, list[float]]] = {
        e: defaultdict(list) for e in EVENTS
    }
    seen_ep: dict[str, dict[str, int]] = {e: defaultdict(int) for e in EVENTS}

    # Chi-square: event -> platform -> emotion -> count
    emo_plat: dict[str, defaultdict[str, defaultdict[str, int]]] = {
        e: defaultdict(lambda: defaultdict(int)) for e in EVENTS
    }

    # Duplicate check on URL (first chunk sample extended by reservoir of urls)
    url_dup_sample: list[str] = []
    MAX_URL_SAMPLE = 80_000

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        on_bad_lines="skip",
    )

    for chunk in reader:
        n_rows += len(chunk)
        if "sentiment" in chunk.columns:
            sentiment_missing += chunk["sentiment"].isna().sum()
        if "url" in chunk.columns:
            url_missing += chunk["url"].isna().sum()

        plat = chunk["url"].map(get_platform) if "url" in chunk.columns else pd.Series(
            ["missing"] * len(chunk)
        )
        platform_counts.update(plat.value_counts().to_dict())

        if "primary_theme" in chunk.columns:
            theme_counts.update(chunk["primary_theme"].fillna("NA").astype(str).value_counts().to_dict())
        if "language" in chunk.columns:
            lang_counts.update(chunk["language"].fillna("NA").astype(str).value_counts().to_dict())

        dt = pd.to_datetime(chunk["date"], utc=True, errors="coerce")
        day = dt.dt.strftime("%Y-%m-%d")
        hour = dt.dt.floor("h").dt.strftime("%Y-%m-%d %H:00")

        s = pd.to_numeric(chunk["sentiment"], errors="coerce")

        # Global daily
        for d, sent in zip(day, s):
            if pd.isna(d):
                continue
            daily_global[d]["n"] += 1
            if pd.notna(sent):
                daily_global[d]["sum_s"] += float(sent)

        blob = (
            chunk["english_keywords"].fillna("").astype(str).str.lower()
            + " "
            + chunk["original_text"].fillna("").astype(str).str.lower()
        )

        emo = chunk["main_emotion"].fillna("NA").astype(str) if "main_emotion" in chunk.columns else pd.Series(
            ["NA"] * len(chunk)
        )
        lang = chunk["language"].fillna("NA").astype(str) if "language" in chunk.columns else pd.Series(
            ["NA"] * len(chunk)
        )

        for ev, spec in EVENTS.items():
            mask = _event_mask(blob, spec["keywords"])
            if not mask.any():
                continue
            sub = mask
            plat_m = plat[sub]
            day_m = day[sub]
            hour_m = hour[sub]
            s_m = s[sub]
            emo_m = emo[sub]
            lang_m = lang[sub]

            for p, d, h, sent, emotion, la in zip(plat_m, day_m, hour_m, s_m, emo_m, lang_m):
                if pd.isna(d):
                    continue
                ed[ev]["day_plat"][d][p] += 1
                ed[ev]["day_lang"][d][la] += 1
                if pd.notna(h):
                    ed[ev]["hour_plat"][h][p] += 1
                ed_daily[ev][d]["n"] += 1
                if pd.notna(sent):
                    ed_daily[ev][d]["sum_s"] += float(sent)
                if pd.notna(h):
                    ed_hourly[ev][h]["n"] += 1
                    if pd.notna(sent):
                        ed_hourly[ev][h]["sum_s"] += float(sent)
                emo_plat[ev][p][emotion] += 1

                # Reservoir sampling for sentiment by platform (skip missing sentiment)
                if pd.isna(sent):
                    continue
                key = p
                seen_ep[ev][key] += 1
                k = seen_ep[ev][key]
                rs = reservoirs[ev][key]
                fv = float(sent)
                if len(rs) < MAX_SENTIMENT_PER_PLATFORM:
                    rs.append(fv)
                else:
                    j = int(rng.integers(0, k))
                    if j < MAX_SENTIMENT_PER_PLATFORM:
                        rs[j] = fv

        # URL sample for duplicate rate
        if "url" in chunk.columns and len(url_dup_sample) < MAX_URL_SAMPLE:
            take = min(MAX_URL_SAMPLE - len(url_dup_sample), len(chunk))
            url_dup_sample.extend(chunk["url"].head(take).dropna().astype(str).tolist())

    dup_rate = 0.0
    if url_dup_sample:
        dup_rate = 1.0 - (len(set(url_dup_sample)) / len(url_dup_sample))

    # DataFrames
    dg = pd.DataFrame(
        [
            {"date": d, "n": v["n"], "mean_sentiment": v["sum_s"] / v["n"] if v["n"] else np.nan}
            for d, v in sorted(daily_global.items())
        ]
    )

    event_n = {e: sum(ed_daily[e][dd]["n"] for dd in ed_daily[e]) for e in EVENTS}
    event_pct = {e: 100.0 * event_n[e] / n_rows if n_rows else 0.0 for e in EVENTS}

    return {
        "n_rows": n_rows,
        "sentiment_missing": int(sentiment_missing),
        "url_missing": int(url_missing),
        "theme_counts": theme_counts,
        "lang_counts": lang_counts,
        "platform_counts": platform_counts,
        "daily_global": dg,
        "ed": ed,
        "ed_daily": ed_daily,
        "ed_hourly": ed_hourly,
        "reservoirs": reservoirs,
        "emo_plat": emo_plat,
        "event_n": event_n,
        "event_pct": event_pct,
        "dup_rate_url_subsample": dup_rate,
        "url_subsample_size": len(url_dup_sample),
    }


def hourly_event_df(ed_hourly: dict[str, dict[str, dict[str, float]]], event: str) -> pd.DataFrame:
    rows = []
    for t, v in sorted(ed_hourly[event].items()):
        n = v["n"]
        mean_s = v["sum_s"] / n if n else np.nan
        rows.append({"hour": t, "n": n, "mean_sentiment": mean_s})
    return pd.DataFrame(rows)


def daily_event_df(ed_daily: dict[str, dict[str, dict[str, float]]], event: str) -> pd.DataFrame:
    rows = []
    for d, v in sorted(ed_daily[event].items()):
        n = v["n"]
        mean_s = v["sum_s"] / n if n else np.nan
        rows.append({"date": d, "n": n, "mean_sentiment": mean_s})
    return pd.DataFrame(rows)


def peak_day_share(ed_daily: dict, event: str) -> float:
    counts = [ed_daily[event][d]["n"] for d in ed_daily[event]]
    if not counts:
        return np.nan
    tot = sum(counts)
    return max(counts) / tot if tot else np.nan


def daily_entropy(ed_daily: dict, event: str) -> float:
    counts = np.array([ed_daily[event][d]["n"] for d in sorted(ed_daily[event].keys())], dtype=float)
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


def long_format_day_platform(ed: dict, event: str) -> pd.DataFrame:
    rows = []
    for d in sorted(ed[event]["day_plat"].keys()):
        for plat, c in ed[event]["day_plat"][d].items():
            rows.append({"date": d, "platform": plat, "n": c})
    return pd.DataFrame(rows)


def long_format_day_language(ed: dict, event: str, top_k: int = TOP_LANG_PLOT) -> pd.DataFrame:
    # aggregate lang totals for event to pick top languages
    tot: Counter[str] = Counter()
    for d in ed[event]["day_lang"]:
        for la, c in ed[event]["day_lang"][d].items():
            tot[la] += c
    keep = {x for x, _ in tot.most_common(top_k)}

    rows = []
    for d in sorted(ed[event]["day_lang"].keys()):
        for la, c in ed[event]["day_lang"][d].items():
            if la in keep:
                rows.append({"date": d, "language": la, "n": c})
    return pd.DataFrame(rows)


def kruskal_epsilon_squared(H: float, n: int, k: int) -> float:
    """Effect size for Kruskal–Wallis (epsilon^2)."""
    if n <= k:
        return np.nan
    return max(0.0, (H - k + 1) / (n - k))


def run_kruskal_for_event(
    reservoirs: dict[str, dict[str, list[float]]], event: str, min_n: int = 30
) -> dict[str, Any]:
    r = reservoirs[event]
    groups = []
    labels = []
    for plat, vals in r.items():
        arr = np.array([x for x in vals if pd.notna(x)], dtype=float)
        if len(arr) >= min_n:
            groups.append(arr)
            labels.append(plat)
    if len(groups) < 2:
        return {"ok": False, "reason": "fewer than 2 platforms with enough rows"}
    stat, p_value = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = kruskal_epsilon_squared(stat, n, k)
    return {
        "ok": True,
        "H_statistic": stat,
        "p_value": p_value,
        "epsilon_sq": eps2,
        "n": n,
        "k_groups": k,
        "platforms": labels,
    }


def collapse_emotion(em: str) -> str:
    if em in ("NA", "nan", ""):
        return "neutral"
    return em


def chi_square_emotion_platform(
    emo_plat: dict[str, defaultdict[str, defaultdict[str, int]]],
    event: str,
    top_platforms: int = 6,
    top_emotions: int = 8,
) -> dict[str, Any]:
    ep = emo_plat[event]
    plat_tot: Counter[str] = Counter()
    for p, ed in ep.items():
        plat_tot[p] += sum(ed.values())
    plat_keep = [x for x, _ in plat_tot.most_common(top_platforms)]

    emo_tot: Counter[str] = Counter()
    for p in plat_keep:
        for em, c in ep[p].items():
            emo_tot[collapse_emotion(em)] += c
    emo_keep = [x for x, _ in emo_tot.most_common(top_emotions)]

    rows = []
    for em in emo_keep:
        row = []
        for p in plat_keep:
            s = 0
            for raw_e, c in ep[p].items():
                if collapse_emotion(raw_e) == em:
                    s += c
            row.append(s)
        rows.append(row)
    table = np.array(rows, dtype=float)
    # merge sparse rows
    if table.size == 0:
        return {"ok": False, "reason": "empty table"}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    # Cramér's V
    n = table.sum()
    r, c = table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1))) if n > 0 and min(r, c) > 1 else np.nan
    return {
        "ok": True,
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": cramers_v,
        "platforms": plat_keep,
        "emotions": emo_keep,
        "table": table,
    }


def build_pca_matrix(
    ed: dict,
    ed_daily: dict,
    ed_hourly: dict,
    event: str,
    top_platforms: int = 6,
) -> tuple[np.ndarray, list[str], list[str], str]:
    """Rows = time buckets; features = counts per top platform + total n + mean sentiment.

    If there are fewer than 7 distinct calendar days with posts (common in sparse
    event slices), switches to **hourly** (UTC) buckets so PCA has enough rows.
    """
    days = sorted(set(ed_daily[event].keys()) | set(ed[event]["day_plat"].keys()))
    use_hourly = len(days) < 7 and len(ed_hourly[event]) >= 4

    plat_tot: Counter[str] = Counter()
    if use_hourly:
        for h in ed[event]["hour_plat"]:
            for p, c in ed[event]["hour_plat"][h].items():
                plat_tot[p] += c
    else:
        for d in ed[event]["day_plat"]:
            for p, c in ed[event]["day_plat"][d].items():
                plat_tot[p] += c
    plats = [x for x, _ in plat_tot.most_common(top_platforms)]

    feats = [f"n_{p}" for p in plats] + ["total_n", "mean_sentiment"]
    X = []
    times: list[str] = []
    gran = "hour" if use_hourly else "day"

    if use_hourly:
        for t in sorted(ed_hourly[event].keys()):
            row = []
            tot = ed_hourly[event].get(t, {"n": 0, "sum_s": 0.0})
            n_b = tot["n"]
            mean_s = tot["sum_s"] / n_b if n_b else 0.0
            for p in plats:
                row.append(float(ed[event]["hour_plat"][t].get(p, 0)))
            row.append(float(n_b))
            row.append(float(mean_s))
            X.append(row)
            times.append(t)
    else:
        for d in days:
            row = []
            tot = ed_daily[event].get(d, {"n": 0, "sum_s": 0.0})
            n_day = tot["n"]
            mean_s = tot["sum_s"] / n_day if n_day else 0.0
            for p in plats:
                row.append(float(ed[event]["day_plat"][d].get(p, 0)))
            row.append(float(n_day))
            row.append(float(mean_s))
            X.append(row)
            times.append(d)

    return np.array(X, dtype=float), times, feats, gran


def reservoirs_to_long_df(reservoirs: dict[str, dict[str, list[float]]], event: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for plat, vals in reservoirs[event].items():
        for v in vals:
            if pd.notna(v):
                rows.append({"platform": plat, "sentiment": float(v)})
    return pd.DataFrame(rows)


def run_pca_kmeans(
    ed: dict,
    ed_daily: dict,
    ed_hourly: dict,
    event: str,
    n_components: int = 2,
    k_range: tuple[int, int] = (2, 5),
) -> dict[str, Any]:
    X, times, feat_names, gran = build_pca_matrix(ed, ed_daily, ed_hourly, event)
    if len(X) < 4:
        return {"ok": False, "reason": f"not enough time buckets ({gran})"}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    Z = pca.fit_transform(Xs)
    sil = []
    models = []
    k_lo, k_hi = k_range
    for k in range(k_lo, min(k_hi + 1, len(X))):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(Z[:, : min(2, Z.shape[1])])
        if len(set(labels)) < 2:
            sil.append(-1.0)
            models.append(None)
            continue
        s = silhouette_score(Z[:, : min(2, Z.shape[1])], labels)
        sil.append(s)
        models.append(km)
    best_i = int(np.argmax(sil))
    best_k = k_lo + best_i
    best_model = models[best_i] if models[best_i] is not None else None
    return {
        "ok": True,
        "Z": Z,
        "days": times,
        "time_granularity": gran,
        "feat_names": feat_names,
        "pca": pca,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "silhouette_scores": list(zip(range(k_lo, k_lo + len(sil)), sil)),
        "best_k": best_k,
        "best_kmeans": best_model,
        "X_scaled": Xs,
    }
