# NanoBEIR Serbian Translation Annotation

Pipeline for loading, annotating, and exporting Serbian machine translations of the [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6) benchmarks using [Argilla](https://argilla.io/).

The goal is to produce a human-verified ground-truth dataset for training and evaluating Serbian text encoder models. Translations were generated with DeepSeek-V3 and are reviewed by human annotators in Argilla.

**Argilla Space:** https://serbian-ai-society-argilla-annotation.hf.space  
**Annotation guide (for annotators):** https://docs.google.com/document/d/1XDhdm9bF8habXG9xE-_WlXXy6-BLee1m2gLqQGrU8Mg/edit?tab=t.uzfx1koe2s3b

---

## Overview

- **13 benchmarks**, 2,811 records total (650 queries + 2,161 passages)
- All records live in a **single merged Argilla dataset** (`NanoBEIR-sr`), shuffled so annotators see a random mix from all benchmarks
- Each record shows the English original and the Serbian machine translation; annotators rate quality (1–5) and correct the translation in-place
- A separate **calibration dataset** (`NanoBEIR-sr-calibration`, 26 records) is annotated by every annotator to measure inter-annotator agreement
- English source texts are cached locally in `.beir_cache/` — no large downloads needed after the first run

---

## Setup

```bash
git clone https://github.com/MihailoPesic/annotation_SAIS.git
cd annotation_SAIS

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

Set credentials (do not commit these):

```bash
export ARGILLA_API_URL="https://serbian-ai-society-argilla-annotation.hf.space"
export ARGILLA_API_KEY="<owner API key — Argilla UI → avatar → My Settings → API Key>"
```

The owner API key is also stored as the `PASSWORD` secret in the HuggingFace Space settings.

---

## Datasets

| Dataset | Purpose | Records | Distribution |
|---|---|---|---|
| `NanoBEIR-sr` | Main annotation task | 2,811 | 1 annotator per record |
| `NanoBEIR-sr-calibration` | Inter-annotator agreement | 26 | Every annotator does all 26 |

The calibration dataset uses `min_submitted=100` so records never auto-complete — every annotator always sees all 26 records regardless of when they join.

---

## Script Reference

### `manage_annotators.py` — Add / remove annotators

Annotators sign in at the Space URL with their HuggingFace account. After first login their account exists in Argilla, but they need to be added to the workspace before they can see any datasets.

```bash
python manage_annotators.py list
python manage_annotators.py add hf_username1 hf_username2
python manage_annotators.py remove hf_username
```

### `check_progress.py` — Monitor annotation progress

Per-benchmark completion table + per-annotator score distribution.

```bash
python check_progress.py
```

### `generate_report.py` — Full HTML dashboard

Self-contained HTML file with summary cards, benchmark progress bars, per-annotator score charts, and a full sortable/filterable annotations table with expandable rows. Opens in the browser automatically.

```bash
python generate_report.py
python generate_report.py --output report.html --no-open
```

### `fix_calibration_discards.py` — Restore accidentally discarded calibration records ⚠️

**When to use:** An annotator presses Discard on a calibration record instead of Submit. The record disappears from their queue. This script detects and reverses it by deleting the discarded response — the record immediately reappears as pending for that annotator.

```bash
# Preview: show who has discards, no changes made
python fix_calibration_discards.py

# Fix: delete the discarded responses and restore the records
python fix_calibration_discards.py --fix
```

Run this whenever an annotator says their calibration queue has fewer than 26 records.

### `reopen_annotation.py` — Re-open a badly annotated main-dataset record

If a submitted annotation is wrong, deleting it returns the record to pending so it can be re-annotated.

```bash
# Preview all responses for a record
python reopen_annotation.py --record-id NanoArguAna_query_42

# Delete one annotator's response
python reopen_annotation.py --record-id NanoArguAna_query_42 --annotator marko_petrovic --fix

# Delete all responses (re-opens for everyone)
python reopen_annotation.py --record-id NanoArguAna_query_42 --all --fix
```

### `compute_agreement.py` — Inter-annotator agreement (quadratic weighted Cohen's kappa)

Run after all annotators have finished the calibration dataset. Computes pairwise Cohen's weighted kappa for every annotator pair.

```bash
python compute_agreement.py
```

Target kappa ≥ 0.6 indicates good agreement. If agreement is low, review annotations together before starting the main dataset.

### `export_annotations_V2.py` — Export completed annotations

```bash
# Push to HuggingFace Hub
python export_annotations_V2.py --to-hub Serbian-AI-Society/nanobeir-annotations

# Save locally as JSONL (completed records only)
python export_annotations_V2.py --to-jsonl annotations.jsonl

# Only records where a real correction was made
python export_annotations_V2.py --to-jsonl annotations.jsonl --require-correction

# Only high-quality annotations (score ≥ 3)
python export_annotations_V2.py --to-jsonl annotations.jsonl --min-score 3
```

### `load_nanobeir.py` — (Re)create the main dataset

Normally run once. Use `--recreate` to delete and rebuild if the schema needs to change.

```bash
python load_nanobeir.py             # create from scratch
python load_nanobeir.py --recreate  # delete existing and rebuild
python load_nanobeir.py --only NanoArguAna  # upsert one benchmark
python load_nanobeir.py --dry-run   # print counts, no upload
```

### `create_calibration_set.py` — (Re)create the calibration dataset

Samples 1 query + 1 passage per benchmark from pending main-dataset records.

```bash
# Delete existing calibration dataset first (via API or Argilla UI), then:
python create_calibration_set.py --min-submitted 100
```

### `check_uniqueness.py` — Verify no duplicate records

Offline check reconstructing all records and verifying uniqueness of IDs, EN text, SR text, and EN+SR pairs.

```bash
python check_uniqueness.py
python check_uniqueness.py --show-dupes
```

---

## Common Runbooks

### Onboarding a new annotator

1. Ask them to open the Space URL and log in with their HuggingFace account (creates their Argilla account).
2. Add them to the workspace:
   ```bash
   python manage_annotators.py add their_hf_username
   ```
3. They now see both datasets. Ask them to complete the **calibration dataset first** (all 26 records) before starting the main dataset.
4. Remind them: **do not use the Discard button in the calibration dataset.**

### An annotator says their calibration queue is short

They accidentally discarded one or more records. Run:

```bash
python fix_calibration_discards.py        # see who is affected and which records
python fix_calibration_discards.py --fix  # restore the records to their queue
```

### After all annotators finish calibration

```bash
python compute_agreement.py
```

Review the pairwise kappa scores. If agreement is acceptable (≥ 0.6), annotators can proceed with the main dataset.

### Regular progress check

```bash
python check_progress.py   # quick table in terminal
python generate_report.py  # full HTML dashboard
```

### Exporting the final dataset

```bash
python export_annotations_V2.py --to-hub Serbian-AI-Society/NanoBEIR-sr-annotated
```

---

## Dataset Structure

Each Argilla record contains:

| Field | Description |
|-------|-------------|
| `annotation_guide_link` | Clickable link to the annotation guide (shown at top of left panel) |
| `source_text_en` | Original English text (query or passage) |
| `translated_text_sr` | DeepSeek-V3 Serbian machine translation |

Annotators fill in (right panel):

| Question | Description |
|----------|-------------|
| `quality_score` | 1–5 rating (1 = completely wrong, 5 = excellent) |
| `comment` | Brief explanation of changes, or "No corrections needed" |
| `corrected_text_sr` | Pre-filled with machine translation; edit in-place, or delete and type "No corrections" |

Metadata (filterable in Argilla):

| Field | Values |
|-------|--------|
| `benchmark` | NanoArguAna, NanoMSMARCO, NanoSciFact, … |
| `record_type` | `query` or `passage` |
| `task_id` | Original ID in the source benchmark |

---

## Benchmarks

| Benchmark | Source | Records |
|-----------|--------|---------|
| NanoArguAna | BeIR/arguana | 100 |
| NanoMSMARCO | BeIR/msmarco | 100 |
| NanoSciFact | BeIR/scifact | 105 |
| NanoFEVER | BeIR/fever | 107 |
| NanoNQ | BeIR/nq | 107 |
| NanoQuoraRetrieval | BeIR/quora | 120 |
| NanoHotpotQA | BeIR/hotpotqa | 150 |
| NanoClimateFEVER | BeIR/climate-fever | 165 |
| NanoFiQA2018 | BeIR/fiqa | 168 |
| NanoSCIDOCS | BeIR/scidocs | 286 |
| NanoNFCorpus | BeIR/nfcorpus | 426 |
| NanoDBPedia | BeIR/dbpedia-entity | 460 |
| NanoTouche2020 | BeIR/webis-touche2020 | 517 |
| **Total** | | **2,811** |

---

## Important Notes

- **Do not use Discard in the calibration dataset.** Every annotator must complete all 26 records. If a discard happens, run `fix_calibration_discards.py --fix`.
- **Do not delete `.beir_cache/`.** It holds cached EN texts; without it, `load_nanobeir.py` re-downloads multi-GB Parquet files.
- **Phantom corrections:** The correction field is pre-filled with the machine translation. If an annotator submits without editing it, the export and report automatically detect this (correction == original translation) and treat it as "no correction made".
