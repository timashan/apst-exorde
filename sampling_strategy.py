import pandas as pd
import random
from datasets import load_dataset

TARGET_N = 1_000_000
SEED = 42
random.seed(SEED)

# -------------------------------
# Date range
# -------------------------------
DATES = pd.date_range("2024-12-01", "2024-12-07").strftime("%Y-%m-%d").tolist()

# -------------------------------
# Platform sizes (global)
# -------------------------------
PLATFORM_SIZES = {
    "x.com": 49_893_080,
    "reddit.com": 14_385_495,
    "bsky.app": 7_292_262,
    "youtube.com": 1_778_020,
    "4channel.org": 233_134,
    "jeuxvideo.com": 61_373,
    "forocoches.com": 55_683,
    "mastodon.social": 54_932,
    "news.ycombinator.com": 32_216,
    "investing.com": 29_090,
}

total_platform = sum(PLATFORM_SIZES.values())

# -------------------------------
# Allocation per date & platform
# -------------------------------
# Step 1: Uniform allocation per date
target_per_date = TARGET_N // len(DATES)
alloc = {}

for d in DATES:
    # Step 2: Within each date, allocate by global platform proportion
    alloc[d] = {p: int(size / total_platform * target_per_date) for p, size in PLATFORM_SIZES.items()}
    # Fix rounding error
    alloc[d][max(alloc[d], key=alloc[d].get)] += target_per_date - sum(alloc[d].values())

print("Sample target per date & platform (first date shown):")
print(alloc[DATES[0]])

# -------------------------------
# Stratified sampler
# -------------------------------
def stratified_sample(df, remaining_alloc):
    samples = []

    for date, platform_alloc in remaining_alloc.items():
        df_date = df[df["day"] == date]
        if df_date.empty:
            continue

        for platform, n_h in platform_alloc.items():
            if n_h <= 0:
                continue
            subset = df_date[df_date["url"].str.contains(platform, na=False)]
            if len(subset) == 0:
                continue

            take_n = min(len(subset), n_h)
            sampled = subset.sample(n=take_n, random_state=SEED)
            samples.append(sampled)

            remaining_alloc[date][platform] -= take_n

    return pd.concat(samples) if samples else pd.DataFrame()

# -------------------------------
# Main sampling with random chunks
# -------------------------------
def run_colab_stratified(max_rows_per_chunk=2_000_000):
    ds = load_dataset("Exorde/exorde-social-media-december-2024-week1", split="train")

    remaining = {d: platform.copy() for d, platform in alloc.items()}
    all_samples = []

    dataset_size = len(ds)
    possible_starts = list(range(0, dataset_size, max_rows_per_chunk))
    random.shuffle(possible_starts)

    for start in possible_starts:
        if sum(sum(v.values()) for v in remaining.values()) <= 0:
            break

        end = min(start + max_rows_per_chunk, dataset_size)
        df_chunk = pd.DataFrame(ds[start:end])

        # Extract date
        df_chunk["day"] = pd.to_datetime(df_chunk["date"]).dt.strftime("%Y-%m-%d")
        # Local shuffle
        df_chunk = df_chunk.sample(frac=1, random_state=SEED).reset_index(drop=True)

        sampled_chunk = stratified_sample(df_chunk, remaining)

        if not sampled_chunk.empty:
            all_samples.append(sampled_chunk)

        total_sampled = TARGET_N - sum(sum(v.values()) for v in remaining.values())
        print(f"Progress: {total_sampled:,} / {TARGET_N:,}")

    final_sample = pd.concat(all_samples).reset_index(drop=True)
    print("\nFinal sample size:", len(final_sample))

    return final_sample

# -------------------------------
# Run
# -------------------------------
sample_df = run_colab_stratified()
sample_df.head()