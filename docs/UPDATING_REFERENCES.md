# Updating the reference sets

The exemplar JSONs in `reference/` are the only files you should need to
edit to change how the pipeline scores Schwartz values or MFT foundations.
Everything in `scripts/` reads from them at runtime - no code changes
required for content updates.

## Files

- `reference/schwartz_exemplars.json` - 9 values, 50 sentences each (max pool size)
- `reference/mft_exemplars.json` - 5 foundations, 25 virtue + 25 vice each (max pool size)

## Common tasks

### Add or remove a single exemplar

Open the JSON, find the value/foundation, edit the relevant `exemplars`
array. Save. Rerun:

```bash
python scripts/03_score_schwartz.py    # for Schwartz changes
python scripts/04_score_mft.py         # for MFT changes
python scripts/05_build_profiles.py
```

You do not need to rerun `02_embed_sentences.py` - sentence embeddings
are cached and reused.

### Add a new Schwartz value (e.g., Hedonism)

1. Open `reference/schwartz_exemplars.json`.
2. Add a new key under `values`:

```json
"Hedonism": {
    "definition": "Pleasure and sensuous gratification for oneself.",
    "motivational_goal": "Seek personal pleasure and enjoyment.",
    "exemplars": [
        "First news-style exemplar sentence...",
        "Second news-style exemplar sentence...",
        ...
    ]
}
```

3. Update `metadata.values_included` and edit `metadata.values_excluded`
   if removing from the exclusion list.
4. Rerun `03_score_schwartz.py` then `05_build_profiles.py`.

The pipeline auto-discovers values from the JSON keys - no code changes
needed. All downstream files and figures will pick up the new value
automatically.

### Add a new MFT foundation (e.g., Liberty)

1. Open `reference/mft_exemplars.json`.
2. Add a new key under `foundations` with the same structure as existing
   foundations (definition, virtue_pole, vice_pole, exemplars.virtue,
   exemplars.vice).
3. Update `metadata.foundations_included`.
4. Rerun `04_score_mft.py` then `05_build_profiles.py`.

### Change the active exemplar size

Edit `scripts/utils.py`:

```python
ACTIVE_SCHWARTZ_SIZE = <int>   # max 50 (or your pool size)
ACTIVE_MFT_SIZE = <int>        # max 50 (or your pool size)
```

`ACTIVE_MFT_SIZE` is split evenly between virtue and vice poles at
runtime, so set it to an even number.

If you change pool sizes substantially, rerun the stability test
(`scripts/06_exemplar_stability_test.py`) to confirm the chosen size
still produces stable rankings.

## Writing good exemplars

The pipeline averages exemplar embeddings to form a category centroid.
What works:

- **News-style prose**, not abstract definitions ("The president ordered
  the military to restore order" - not "Authority is respect for
  hierarchy").
- **One value or foundation per sentence.** A sentence that mixes two
  is at best a wash and at worst pulls the centroid toward another category.
- **Concrete actors and actions** ("the cardinal blessed the faithful",
  "the senator chaired the committee") rather than abstract nouns.
- **Topical variety** within each category - drawing every Security
  exemplar from typhoons will tilt the centroid toward weather-related
  language.

What doesn't work:

- Lists of keywords.
- Sentences that explicitly name the category being scored ("The story
  emphasizes care for the victims" tilts toward meta-commentary, not
  framing).
- Direct quotes from the corpus you're going to score - this leaks the
  target into the reference and inflates similarity scores artificially.

## After editing

1. Rerun the affected scoring script + `05_build_profiles.py`.
2. (Recommended) Rerun `06_exemplar_stability_test.py` if you changed
   the pool size by more than ~5 sentences per category.
3. Check `outputs/diagnostics/exemplar_coverage.csv` - if a category has
   near-zero dominant counts across all outlets, its exemplars may not
   be distinguishing well from neighbors.
4. Skim `outputs/profiles/comparison_*.csv` to confirm changes make sense
   given the edit.

## Reverting

The reference JSONs are version-controlled (git). To undo any edit:

```bash
git diff reference/                    # see what changed
git checkout reference/                # discard all reference changes
```
