# Argilla Translation Annotation Setup

Scripts for setting up and managing the EN→SR translation annotation environment on Argilla v2.

## Prerequisites

- Python 3.10+
- An Argilla v2.x server running (e.g. on HuggingFace Spaces)
- `pip install argilla`

## Quick Start

```bash
# Set credentials once
export ARGILLA_API_URL="https://serbian-ai-society-argilla-annotation.hf.space"
export ARGILLA_API_KEY="your-api-key"

# 1. Create the dataset schema
python setup_dataset.py

# 2. Load sample data for testing
python load_data.py --sample 5

# 3. (Annotators do their work in the Argilla UI)

# 4. Export completed annotations as JSONL
python export_annotations.py --output annotations.jsonl

# 5. Check progress
python analytics.py
```

## Loading Real Data

Prepare a JSONL file where each line looks like:

```json
{"id": "rec_0001", "source_text_en": "The court ruled...", "translated_text_sr": "Sud je presudio..."}
```

Then run:

```bash
python load_data.py --input your_data.jsonl --source-dataset "msmarco-batch-1"
```

CSV is also supported (columns: id, source_text_en, translated_text_sr).

## Output Format

The export script produces JSONL with one record per annotation:

```json
{
    "id": "rec_0001",
    "corrected_text_sr": "Sud je doneo presudu...",
    "quality_score": "medium",
    "annotator_id": "marko",
    "timestamp": "2026-04-02T14:30:00Z",
    "edit_metadata": {
        "was_modified": true,
        "edit_distance": 8
    }
}
```

## Workspace Note

If your Argilla is deployed on HF Spaces with HF OAuth sign-in, use `--workspace argilla` (the default). User accounts are managed through HF OAuth, not through the Argilla admin panel.

## Files

| File | Purpose |
|------|---------|
| setup_dataset.py | Creates the dataset schema with fields, questions, guidelines |
| load_data.py | Imports translation pairs from JSONL/CSV or generates samples |
| export_annotations.py | Exports completed annotations as JSONL with edit metadata |
| analytics.py | Reports progress, quality distribution, per-annotator stats |
