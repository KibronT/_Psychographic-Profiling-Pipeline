# Limitations

What this pipeline does *not* do, and why.

## Methodological

### Single-label aggregation by default
Every sentence gets one dominant Schwartz value and one dominant MFT
foundation. A sentence about a church-led rally over drug-war casualties
could legitimately read as Security, Universalism, *and* Tradition, but
only the top-scoring value gets credit in the `pct_dominant` numbers.

**Mitigation already in place:** the full per-sentence cosine vector for
all 9 values + 5 foundations is preserved in
`outputs/profiles/sentence_scores.parquet`. Researchers can re-aggregate
with soft assignment (fractional weights), multi-label thresholding, or
top-K rules without rerunning embeddings.

### Exemplar subjectivity
The reference sentences were drafted by hand from Schwartz (1992, 2012)
and Haidt & Graham (2007) definitions. They embed the author's choices
about what counts as a clear instance of each category - particularly for
Philippine context. Two researchers writing the same number of exemplars
would produce overlapping but non-identical centroids.

**Mitigation:** the stability test confirms ranking stability across
random subsamples of the pool, and the JSONs are designed to be edited.
See [UPDATING_REFERENCES.md](UPDATING_REFERENCES.md).

### Median salience filter is symmetric, not absolute
We keep the top 50% of sentences by salience *per outlet*. The absolute
cosine cutoff varies by outlet - an outlet with consistently lower
salience overall will retain sentences with lower scores than one with
higher salience. This is intentional (it prevents the comparison from
collapsing if one outlet has a more diffuse style), but it does mean
"% dominant" numbers compare *relative* prominence within each outlet's
half, not absolute moral intensity.

### Hedonism is excluded from Schwartz
Schwartz's canonical model has 10 values; this pipeline scores 9.
Hedonism (pleasure, sensuous gratification for oneself) is rarely the
surface motivation of news content. Stimulation is the closest
functional analogue and is included. Researchers who want to score
Hedonism explicitly can append a new key to `schwartz_exemplars.json`
and rerun the pipeline.

## Linguistic

### English-Tagalog code-mixing not separately handled
SBERT (`all-MiniLM-L6-v2`) is multilingual-aware but not Philippine-tuned.
Tagalog and Taglish sentences are embedded with the same model and
compared to English exemplars. This works because the model learned
cross-lingual alignments during training, but performance has not been
audited for Tagalog-heavy passages.

**Possible improvement:** swap in a multilingual model
(`paraphrase-multilingual-MiniLM-L12-v2`) or a Filipino-fine-tuned
encoder, then redraft exemplars in Tagalog and Taglish to broaden the
centroids.

### Exemplars are mostly Philippine-context
Roughly two-thirds of the drafted exemplars use Philippine-specific
context (Marcos, Malacañang, Black Nazarene, etc.). This is intentional -
the corpus is Philippine news - but it could bias the centroids toward
matching Philippine surface features rather than abstract value content.
The stability test treats this as a fixed property of the pool; a
research direction is to compare against a generic-news exemplar set.

### Sentence splitter is a simple regex
The splitter uses `[.!?] + whitespace + capital-letter-or-quote`. It
correctly handles ~95% of news prose but will miss occasional
abbreviations and split-quotation cases. Errors are sentence-local; they
do not affect article boundaries or aggregation.

## Temporal and content scope

### No temporal analysis
Article publication dates are not used at any stage. Profiles are computed
over the entire 2024 corpus per outlet. Dates in the source JSONL are
present but unreliable across the three outlets (different formats,
some scraped from page metadata, some from URL slugs) and have not been
normalized. Temporal trends are out of scope for this pipeline.

### Outlet sample sizes are unequal
ABS-CBN (397 articles) is roughly a third the size of GMA (1,216) and
Rappler (1,178). The pipeline computes per-outlet proportions, so the
comparisons are normalized - but the smaller ABS-CBN sample produces
noisier per-category counts. The exemplar coverage CSV in
`outputs/diagnostics/` flags categories with low absolute counts.

## What the profiles do *not* directly measure

### Reader/audience psychographics
The profiles characterize the **psychological orientation projected by
the outlet's content**, not the personalities of individual readers.
The inferential bridge - that audiences self-select into media matching
their orientation (selective exposure theory) - is well-established but
not tested here.

### Editorial intent vs. surface content
We measure what the words say, not what the editors meant. A sentence
reporting on a violent crime gets scored as Care/vice regardless of
whether the editorial framing is critical, neutral, or sensational.
Sentiment-level nuance is outside the moral-framing layer.
