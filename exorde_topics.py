"""
Unsupervised topics per keyword-defined event: NMF on TF–IDF (short social text).

Use for exploration; topics are not ground truth. See `topic_modeling_sensitivity.ipynb`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from exorde_analysis import EVENTS, event_keyword_mask, get_platform, text_blob_series

_URL_RE = re.compile(r"https?://\S+")

# Default row cap so a full pass over all events stays interactive on a laptop.
DEFAULT_MAX_DOCS = 20_000


def _strip_urls(s: str) -> str:
    return _URL_RE.sub(" ", s)


def _topic_entropy(W: np.ndarray) -> np.ndarray:
    rs = W.sum(axis=1, keepdims=True) + 1e-15
    P = W / rs
    return -(P * np.log(P + 1e-15)).sum(axis=1)


def _build_doc_topics_df(
    sub: pd.DataFrame,
    event_key: str,
    W: np.ndarray,
    *,
    events: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """One row per post in the fitted subsample; index = original dataframe index."""
    topic_id = np.argmax(W, axis=1).astype(np.int64)
    topic_weight = W.max(axis=1)
    ent = _topic_entropy(W)

    plat = (
        sub["url"].map(get_platform)
        if "url" in sub.columns
        else pd.Series(["missing"] * len(sub), index=sub.index)
    )
    dt = pd.to_datetime(sub["date"], utc=True, errors="coerce") if "date" in sub.columns else None

    rows: dict[str, Any] = {
        "event_key": event_key,
        "event_label": events[event_key]["label"],
        "topic_id": topic_id,
        "topic_weight": topic_weight,
        "topic_entropy": ent,
        "platform": plat.values,
    }
    if dt is not None:
        rows["date"] = dt.values
        rows["date_str"] = dt.dt.strftime("%Y-%m-%d").values
    if "sentiment" in sub.columns:
        rows["sentiment"] = pd.to_numeric(sub["sentiment"], errors="coerce").values
    if "language" in sub.columns:
        rows["language"] = sub["language"].astype(str).values
    if "main_emotion" in sub.columns:
        rows["main_emotion"] = sub["main_emotion"].fillna("NA").astype(str).values
    if "primary_theme" in sub.columns:
        rows["primary_theme"] = sub["primary_theme"].fillna("NA").astype(str).values

    out = pd.DataFrame(rows, index=sub.index)
    out.index.name = "source_index"
    return out


def run_nmf_topics(
    df: pd.DataFrame,
    event_key: str,
    *,
    events: dict[str, dict[str, Any]] | None = None,
    n_topics: int = 10,
    max_docs: int = DEFAULT_MAX_DOCS,
    min_chars: int = 15,
    language: str | None = None,
    random_state: int = 42,
    top_terms: int = 12,
    return_doc_topics: bool = False,
) -> dict[str, Any]:
    """
    Subset rows to `event_key`, optionally filter by `language` (exact match on `df['language']`).

    Fits NMF on TF–IDF bag-of-words (English stop words removed). Returns top terms per topic
    and prevalence (argmax document-topic weight).

    Pass `events` to match the same keyword dict used elsewhere (default: baseline `EVENTS`).

    When `return_doc_topics=True`, also returns `doc_topics` — a DataFrame aligned to the fitted
    posts (original index) with `topic_id`, `topic_weight`, `topic_entropy`, and metadata columns
    for downstream plots (platform, date, sentiment, etc.).
    """
    evs = EVENTS if events is None else events
    if event_key not in evs:
        return {"ok": False, "reason": f"unknown event_key: {event_key!r}"}

    sub = df.loc[event_keyword_mask(df, event_key, events=evs)].copy()
    if language is not None and "language" in sub.columns:
        sub = sub[sub["language"].astype(str) == language]

    blob = text_blob_series(sub).map(_strip_urls).str.replace(r"\s+", " ", regex=True).str.strip()
    ok_len = blob.str.len() >= min_chars
    sub = sub.loc[ok_len]
    blob = blob.loc[ok_len]

    n = len(sub)
    if n < 80:
        return {
            "ok": False,
            "reason": f"too few posts after filters (n={n}); need at least ~80",
            "n_posts": n,
        }

    rng = np.random.default_rng(random_state)
    if n > max_docs:
        ix = rng.choice(sub.index.values, size=max_docs, replace=False)
        sub = sub.loc[ix]
        blob = text_blob_series(sub).map(_strip_urls).str.replace(r"\s+", " ", regex=True).str.strip()

    texts = blob.tolist()
    n_docs = len(texts)

    n_comp = min(n_topics, n_docs - 1, 50)
    n_comp = max(2, n_comp)

    min_df = max(2, min(5, n_docs // 5000))
    max_features = min(12_000, max(500, n_docs * 2))

    # max_df < 1.0 removes terms that appear in every document; in tight keyword slices many
    # words are ubiquitous, so use 1.0 (only remove terms seen in a strict superset — none).
    vec = TfidfVectorizer(
        max_df=1.0,
        min_df=min_df,
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError as e:
        return {"ok": False, "reason": f"vectorizer failed: {e}", "n_posts": n_docs}

    if X.shape[1] < n_comp:
        n_comp = max(2, min(n_comp, X.shape[1]))

    nmf = NMF(
        n_components=n_comp,
        init="nndsvda",
        random_state=random_state,
        max_iter=400,
        l1_ratio=0.0,
    )
    try:
        W = nmf.fit_transform(X)
    except Exception as e:
        return {"ok": False, "reason": f"NMF failed: {e}", "n_posts": n_docs}

    H = nmf.components_
    feat = vec.get_feature_names_out()
    labels = np.argmax(W, axis=1)
    counts = np.bincount(labels, minlength=n_comp).astype(float)
    prev = counts / counts.sum() if counts.sum() else counts

    topics_out: list[dict[str, Any]] = []
    for j in range(n_comp):
        top_ix = np.argsort(H[j])[-top_terms:][::-1]
        terms = [feat[i] for i in top_ix]
        wts = [float(H[j, i]) for i in top_ix]
        topics_out.append(
            {
                "id": j,
                "prevalence": float(prev[j]),
                "top_terms": terms,
                "weights": wts,
            }
        )

    topics_out.sort(key=lambda t: -t["prevalence"])

    out: dict[str, Any] = {
        "ok": True,
        "event_key": event_key,
        "label": evs[event_key]["label"],
        "n_posts": n_docs,
        "n_topics": n_comp,
        "language_filter": language,
        "topics": topics_out,
    }
    if return_doc_topics:
        out["doc_topics"] = _build_doc_topics_df(sub, event_key, W, events=evs)
    return out


def run_nmf_topics_all_events(
    df: pd.DataFrame,
    *,
    events: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run `run_nmf_topics` for every key in `events` (default: baseline `EVENTS`)."""
    evs = EVENTS if events is None else events
    return {ek: run_nmf_topics(df, ek, events=evs, **kwargs) for ek in evs}
