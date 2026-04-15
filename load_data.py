"""
Load translation pairs into the Argilla annotation dataset.

Accepts JSONL input where each line has:
    {"id": "...", "source_text_en": "...", "translated_text_sr": "..."}

Can also accept CSV with columns: id, source_text_en, translated_text_sr

Usage:
    python load_data.py --input data.jsonl --api-url <URL> --api-key <KEY>
    python load_data.py --input data.csv --api-url <URL> --api-key <KEY>
    python load_data.py --sample 5  # load 5 demo records for testing

Environment variables:
    ARGILLA_API_URL, ARGILLA_API_KEY
"""

import argparse
import csv
import json
import os
import sys

import argilla as rg


# ---------------------------------------------------------------------------
# Sample data for testing the setup
# ---------------------------------------------------------------------------

SAMPLE_DATA = [
    {
        "id": "sample_001",
        "source_text_en": "The Supreme Court ruled that the new legislation violates constitutional rights to privacy.",
        "translated_text_sr": "Vrhovni sud je presudio da novi zakon krši ustavna prava na privatnost.",
    },
    {
        "id": "sample_002",
        "source_text_en": "Machine learning models require large amounts of labeled data for training.",
        "translated_text_sr": "Modeli mašinskog učenja zahtevaju velike količine označenih podataka za obuku.",
    },
    {
        "id": "sample_003",
        "source_text_en": "The patient was diagnosed with a rare autoimmune disorder affecting the nervous system.",
        "translated_text_sr": "Pacijentu je dijagnostikovan redak autoimuni poremećaj koji utiče na nervni sistem.",
    },
    {
        "id": "sample_004",
        "source_text_en": "Investors are concerned about the impact of rising interest rates on the housing market.",
        "translated_text_sr": "Investitori su zabrinuti za uticaj rastućih kamatnih stopa na tržište nekretnina.",
    },
    {
        "id": "sample_005",
        "source_text_en": "The defendant argued that the evidence was obtained through an illegal search and seizure.",
        "translated_text_sr": "Optuženi je tvrdio da su dokazi pribavljeni putem nezakonitog pretresa i oduzimanja.",
    },
]


def load_jsonl(filepath: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_num}, invalid JSON: {e}")
                continue

            # Validate required fields
            if not all(k in obj for k in ("id", "source_text_en", "translated_text_sr")):
                print(f"Warning: skipping line {line_num}, missing required fields")
                continue

            records.append(obj)
    return records


def load_csv(filepath: str) -> list[dict]:
    """Load records from a CSV file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            if not all(k in row for k in ("id", "source_text_en", "translated_text_sr")):
                print(f"Warning: skipping row {row_num}, missing required columns")
                continue
            records.append(dict(row))
    return records


def upload_records(records: list[dict], api_url: str, api_key: str,
                   workspace: str = "argilla",
                   dataset_name: str = "translation-annotation-sr",
                   source_dataset: str = "manual"):
    """Upload records to the Argilla dataset."""

    client = rg.Argilla(api_url=api_url, api_key=api_key)
    dataset = client.datasets(name=dataset_name, workspace=workspace)

    if dataset is None:
        print(f"Error: dataset '{dataset_name}' not found. Run setup_dataset.py first.")
        sys.exit(1)

    argilla_records = []
    for rec in records:
        argilla_records.append(
            rg.Record(
                id=rec["id"],
                fields={
                    "source_text_en": rec["source_text_en"],
                    "translated_text_sr": rec["translated_text_sr"],
                },
                metadata={
                    "task_id": rec["id"],
                    "source_dataset": source_dataset,
                },
            )
        )

    dataset.records.log(argilla_records)
    print(f"Uploaded {len(argilla_records)} records to '{dataset_name}'.")


def main():
    parser = argparse.ArgumentParser(description="Load translation data into Argilla")
    parser.add_argument("--input", help="Path to JSONL or CSV file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Load N sample records for testing (max 5)")
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="translation-annotation-sr")
    parser.add_argument("--source-dataset", default="manual",
                        help="Tag identifying the source of this data batch")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    # Determine data source
    if args.sample > 0:
        n = min(args.sample, len(SAMPLE_DATA))
        records = SAMPLE_DATA[:n]
        print(f"Using {n} sample records for testing.")
    elif args.input:
        if args.input.endswith(".csv"):
            records = load_csv(args.input)
        else:
            records = load_jsonl(args.input)
        print(f"Loaded {len(records)} records from {args.input}")
    else:
        print("Error: provide --input <file> or --sample <N>")
        sys.exit(1)

    if not records:
        print("No records to upload.")
        sys.exit(1)

    upload_records(records, args.api_url, args.api_key,
                   args.workspace, args.dataset_name, args.source_dataset)


if __name__ == "__main__":
    main()
