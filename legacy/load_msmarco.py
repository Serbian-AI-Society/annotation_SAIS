"""
Load Serbian-AI-Society/ms_marco_en-sr translations into Argilla for annotation.

This script pairs English and Serbian texts by query_id and creates
annotation tasks for translation quality review.

You can annotate queries, passages, or both.

Usage:
    # Annotate 100 random query translations
    python load_msmarco.py --type queries --n 100

    # Annotate 100 random passage translations
    python load_msmarco.py --type passages --n 100

    # Annotate 50 of each
    python load_msmarco.py --type both --n 50

    # Use a specific split
    python load_msmarco.py --type queries --n 200 --split train

Prerequisites:
    You may need to log in to HuggingFace first if the dataset is private:
        huggingface-cli login
"""

import argparse
import json
import os
import random
import sys

import argilla as rg
from datasets import load_dataset


def load_query_pairs(split: str, n: int, seed: int = 42) -> list[dict]:
    """Load paired English-Serbian queries from ms_marco_en-sr."""
    print(f"Loading English queries ({split} split)...")
    en_ds = load_dataset("Serbian-AI-Society/ms_marco_en-sr", "v1.1_en", split=split)

    print(f"Loading Serbian queries ({split} split)...")
    sr_ds = load_dataset("Serbian-AI-Society/ms_marco_en-sr", "v1.1_sr", split=split)

    # Build lookup by query_id for Serbian
    sr_by_id = {}
    for row in sr_ds:
        sr_by_id[row["query_id"]] = row

    # Pair them
    pairs = []
    for en_row in en_ds:
        qid = en_row["query_id"]
        sr_row = sr_by_id.get(qid)
        if sr_row is None:
            continue
        pairs.append({
            "id": f"query_{qid}",
            "source_text_en": en_row["query"],
            "translated_text_sr": sr_row["query"],
        })

    # Sample
    random.seed(seed)
    if n < len(pairs):
        pairs = random.sample(pairs, n)

    print(f"Prepared {len(pairs)} query pairs for annotation.")
    return pairs


def load_passage_pairs(split: str, n: int, seed: int = 42) -> list[dict]:
    """Load paired English-Serbian passages from ms_marco_en-sr.

    Each query has multiple passages. We pick one selected passage per query
    (the one marked is_selected=1) when available, otherwise the first passage.
    """
    print(f"Loading English data ({split} split)...")
    en_ds = load_dataset("Serbian-AI-Society/ms_marco_en-sr", "v1.1_en", split=split)

    print(f"Loading Serbian data ({split} split)...")
    sr_ds = load_dataset("Serbian-AI-Society/ms_marco_en-sr", "v1.1_sr", split=split)

    # Build lookup by query_id for Serbian
    sr_by_id = {}
    for row in sr_ds:
        sr_by_id[row["query_id"]] = row

    pairs = []
    for en_row in en_ds:
        qid = en_row["query_id"]
        sr_row = sr_by_id.get(qid)
        if sr_row is None:
            continue

        en_passages = en_row.get("passages", {})
        sr_passages = sr_row.get("passages", {})

        # Handle both dict-of-lists and list-of-dicts formats
        if isinstance(en_passages, dict):
            en_texts = en_passages.get("passage_text", [])
            en_selected = en_passages.get("is_selected", [])
            sr_texts = sr_passages.get("passage_text", [])
        else:
            en_texts = [p.get("passage_text", "") for p in en_passages]
            en_selected = [p.get("is_selected", 0) for p in en_passages]
            sr_texts = [p.get("passage_text", "") for p in sr_passages]

        if not en_texts or not sr_texts:
            continue

        # Pick the selected passage if available
        idx = 0
        for i, sel in enumerate(en_selected):
            if sel == 1:
                idx = i
                break

        if idx >= len(sr_texts):
            idx = 0

        en_text = en_texts[idx] if idx < len(en_texts) else en_texts[0]
        sr_text = sr_texts[idx] if idx < len(sr_texts) else sr_texts[0]

        if not en_text.strip() or not sr_text.strip():
            continue

        pairs.append({
            "id": f"passage_{qid}_{idx}",
            "source_text_en": en_text,
            "translated_text_sr": sr_text,
        })

    random.seed(seed)
    if n < len(pairs):
        pairs = random.sample(pairs, n)

    print(f"Prepared {len(pairs)} passage pairs for annotation.")
    return pairs


def upload_to_argilla(records: list[dict], api_url: str, api_key: str,
                      workspace: str, dataset_name: str, source_tag: str):
    """Upload records to Argilla."""
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
                    "source_dataset": source_tag,
                },
            )
        )

    dataset.records.log(argilla_records)
    print(f"Uploaded {len(argilla_records)} records to '{dataset_name}'.")


def save_jsonl(records: list[dict], path: str):
    """Save records to JSONL file (useful as backup or for offline review)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load ms_marco_en-sr translations into Argilla"
    )
    parser.add_argument("--type", choices=["queries", "passages", "both"],
                        default="queries", help="What to annotate (default: queries)")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples per type (default: 100)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to sample from (default: train)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--save-jsonl", default=None,
                        help="Also save records to a JSONL file")
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="translation-annotation-sr-v2")

    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    all_records = []

    if args.type in ("queries", "both"):
        all_records.extend(load_query_pairs(args.split, args.n, args.seed))

    if args.type in ("passages", "both"):
        all_records.extend(load_passage_pairs(args.split, args.n, args.seed))

    if not all_records:
        print("No records to upload.")
        sys.exit(1)

    if args.save_jsonl:
        save_jsonl(all_records, args.save_jsonl)

    upload_to_argilla(
        all_records, args.api_url, args.api_key,
        args.workspace, args.dataset_name,
        source_tag=f"ms_marco_{args.type}_{args.split}",
    )


if __name__ == "__main__":
    main()