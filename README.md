# apst-exorde

Event-scoped analysis on a platform-stratified sample from [Exorde/exorde-social-media-december-2024-week1](https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1). Core logic is in `exorde_analysis.py`; the notebook `analysis_events.ipynb` drives plots and tables.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended) or Python **3.14+** matching `pyproject.toml`
- The sample CSV (about 1M rows), either at the repo root as `stratified_sample_1M.csv` or under `data/` — see below

## Install

```bash
uv sync
```

## Run the analysis

1. **CSV path** — The notebook sets `CSV_PATH = Path("data/stratified_sample_1M.csv")`.

2. **Start Jupyter** from the repository root so imports resolve:

```bash
uv run jupyter notebook analysis_events.ipynb
```

Use `uv run jupyter lab` if you prefer Lab. Run all cells; the first full pass over the CSV can take a few minutes.

## Optional: regenerate the stratified sample

`sampling_strategy.py` streams the Hugging Face dataset and builds a new 1M-row stratified sample. It is heavy on RAM, disk, and network — mainly intended for Colab or similar. Run only if you need a fresh CSV:

```bash
uv run python sampling_strategy.py
```
