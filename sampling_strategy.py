# ======================================================
# Colab-friendly Stratified Sampling of Exorde Social Media
# ======================================================
# Target: 1,000,000 rows
# Stratified by platform (no time bias)
# Uses HuggingFace dataset directly, shuffling in memory
# ======================================================

import pandas as pd
import random
from datasets import load_dataset

# -------------------------------
# 1. Parameters
# -------------------------------
TARGET_N = 1_000_000
SEED = 42
random.seed(SEED)

# Known top platforms and approximate counts
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

# -------------------------------
# 2. Compute proportional allocation
# -------------------------------
total_known = sum(PLATFORM_SIZES.values())
alloc = {p: int(size / total_known * TARGET_N) for p, size in PLATFORM_SIZES.items()}
diff = TARGET_N - sum(alloc.values())
# Fix rounding difference
alloc[max(alloc, key=alloc.get)] += diff

print("Target sample per platform:")
for k, v in sorted(alloc.items(), key=lambda x: -x[1]):
    print(f"{k:<20} → {v:,}")

# -------------------------------
# 3. Helper: sample stratified from a DataFrame
# -------------------------------
def stratified_sample(df, remaining_alloc):
    """Draw stratified samples from df according to remaining_alloc."""
    samples = []
    for platform, n_h in remaining_alloc.items():
        if n_h <= 0:
            continue
        subset = df[df['url'].apply(lambda u: platform in str(u))]
        if len(subset) > 0:
            take_n = min(len(subset), n_h)
            sampled = subset.sample(n=take_n, random_state=SEED)
            samples.append(sampled)
            remaining_alloc[platform] -= take_n
    return pd.concat(samples) if samples else pd.DataFrame()

# -------------------------------
# 4. Load dataset and sample
# -------------------------------
def run_colab_stratified(target_alloc=alloc, max_rows_per_chunk=5_000_000):
    """
    Memory-efficient stratified sampling from HuggingFace dataset in Colab.
    - Loads chunks of dataset to avoid RAM issues
    - Shuffles each chunk to avoid time bias
    """
    ds = load_dataset("Exorde/exorde-social-media-december-2024-week1", split="train")
    remaining = target_alloc.copy()
    all_samples = []

    # Process in chunks
    start = 0
    while sum(remaining.values()) > 0 and start < len(ds):
        end = min(start + max_rows_per_chunk, len(ds))
        df_chunk = pd.DataFrame(ds[start:end])
        df_chunk = df_chunk.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle rows

        sampled_chunk = stratified_sample(df_chunk, remaining)
        if not sampled_chunk.empty:
            all_samples.append(sampled_chunk)

        print(f"Progress: {TARGET_N - sum(remaining.values()):,} / {TARGET_N:,} sampled")
        start = end

        # Break if all targets reached
        if all(v <= 0 for v in remaining.values()):
            break

    final_sample = pd.concat(all_samples).reset_index(drop=True)
    print("\nFinal sample size:", len(final_sample))
    return final_sample

# -------------------------------
# 5. Run
# -------------------------------
sample_df = run_colab_stratified()
sample_df.head()