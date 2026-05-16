# Exemplar Stability Test Results

## Decision rule
Pick the smallest exemplar size where the mean Spearman rank correlation
between per-outlet rankings at that size and the next-larger size is
>= 0.95. Fallback: use the largest tested size (50) if no size
converges.

## Sizes tested
[15, 25, 35, 50]  (MFT sizes are split evenly between virtue and vice poles.)

## Chosen sizes
- Schwartz: **50** per value
  - No consecutive pair reached mean Spearman >= 0.95; using largest size (50).
- MFT: **50** per foundation
  - No consecutive pair reached mean Spearman >= 0.95; using largest size (50).

## How to apply
Edit `scripts/utils.py` and set:

```python
ACTIVE_SCHWARTZ_SIZE = 50
ACTIVE_MFT_SIZE = 50
```

Then rerun `03_score_schwartz.py`, `04_score_mft.py`, and `05_build_profiles.py`
to produce profiles at the chosen sizes.

## Files
- `stability_results.csv` - full Spearman + top-3 results across all size pairs
- `stability_convergence.png` - visualization of mean Spearman by size pair
- `active_sizes.json` - machine-readable chosen sizes
