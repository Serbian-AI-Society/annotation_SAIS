# NanoBEIR Serbian Translation Annotation

Pipeline for loading, annotating, and exporting Serbian machine translations of the [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6) benchmarks using [Argilla](https://argilla.io/).

The goal is to produce a human-verified ground-truth dataset for training and evaluating Serbian text encoder models. Translations were generated with DeepSeek-V3 and are reviewed by human annotators in Argilla.

## Overview

- **13 benchmarks**, 2,811 records total (650 queries + 2,161 passages)
- All records live in a **single merged Argilla dataset** (`NanoBEIR-sr`), shuffled so annotators see a random mix from all benchmarks
- Each record shows the English original alongside the Serbian machine translation; annotators rate quality (1–5) and optionally correct the translation
- English source texts are cached locally in `.beir_cache/` — no large downloads needed after the first run

## Setup

```bash
# Clone the repo
git clone https://github.com/MihailoPesic/annotation_SAIS.git
cd annotation_SAIS

# Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Set credentials (do not commit these)
export ARGILLA_API_URL="https://serbian-ai-society-argilla-annotation.hf.space"
export ARGILLA_API_KEY="your-api-key"
# On Windows PowerShell: . .\env.ps1  (create env.ps1 from env.ps1.example)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `load_nanobeir.py` | Main pipeline — loads all 13 benchmarks into Argilla |
| `check_progress.py` | Per-benchmark annotation progress dashboard |
| `export_annotations_V2.py` | Export completed annotations to HF Hub or JSONL |
| `setup_dataset_v2.py` | One-off dataset schema creation (called by load_nanobeir.py) |

### Load / reload data

```bash
# Full run — creates NanoBEIR-sr dataset with all 2,811 shuffled records
python load_nanobeir.py

# Add or refresh a single benchmark (upserts into existing dataset)
python load_nanobeir.py --only NanoArguAna

# Dry run — prints record counts without uploading
python load_nanobeir.py --dry-run

# Custom passage cap (default 10 per query)
python load_nanobeir.py --max-pos 5
```

### Check annotation progress

```bash
python check_progress.py
```

Output:
```
Benchmark                         Queries        Passages           Total
                               done/total      done/total      done/total
---------------------------------------------------------------------------
NanoArguAna                    45/50 (90%)     48/50 (96%)    93/100 (93%)
NanoTouche2020                  12/50 (24%)    89/467 (19%)  101/517 (20%)
...
TOTAL                                                       194/2811 (7%)
```

### Export annotations

```bash
# Push to HuggingFace Hub
python export_annotations_V2.py --to-hub Serbian-AI-Society/nanobeir-annotations

# Save locally as JSONL
python export_annotations_V2.py --to-jsonl annotations.jsonl
```

## Dataset structure

Each Argilla record contains:

| Field | Description |
|-------|-------------|
| `source_text_en` | Original English text (query or passage) |
| `translated_text_sr` | DeepSeek-V3 Serbian machine translation |

Annotators fill in:

| Question | Description |
|----------|-------------|
| `quality_score` | 1–5 rating (1 = completely wrong, 5 = excellent) |
| `corrected_text_sr` | Corrected Serbian translation, or "No corrections" |
| `comment` | Brief explanation of changes, or "No corrections needed" |

Metadata (filterable in Argilla):

| Field | Values |
|-------|--------|
| `benchmark` | NanoArguAna, NanoMSMARCO, NanoSciFact, … |
| `record_type` | `query` or `passage` |
| `task_id` | Original ID in the source benchmark |

## Benchmarks and record counts

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

## Cache

`.beir_cache/` contains JSON files with English source texts fetched from BeIR corpora. Five of the 13 benchmarks use corpora with 4–9 million rows; fetching them requires downloading multi-GB Parquet files on the first run. The cache avoids re-downloading on subsequent runs. **Do not delete `.beir_cache/`.**

## Argilla instance

Hosted on HuggingFace Spaces: `https://serbian-ai-society-argilla-annotation.hf.space`
