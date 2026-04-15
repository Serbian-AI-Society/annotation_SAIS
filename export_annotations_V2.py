"""
Export annotations using Argilla's native export methods.

Usage:
    # Push to HuggingFace Hub (recommended)
    python export_v2.py --to-hub Serbian-AI-Society/translation-annotations

    # Save to local disk
    python export_v2.py --to-disk ./exported_annotations

    # Export as JSONL (custom, for compatibility)
    python export_v2.py --to-jsonl annotations.jsonl
"""

import argparse
import json
import os
import sys

import argilla as rg


def main():
    parser = argparse.ArgumentParser(description="Export annotations")
    parser.add_argument("--to-hub", help="Push to HF Hub (e.g. Serbian-AI-Society/translation-annotations)")
    parser.add_argument("--to-disk", help="Save to local directory")
    parser.add_argument("--to-jsonl", help="Export as JSONL file")
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="NanoBEIR-sr")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required")
        sys.exit(1)

    if not any([args.to_hub, args.to_disk, args.to_jsonl]):
        print("Error: specify at least one of --to-hub, --to-disk, or --to-jsonl")
        sys.exit(1)

    client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)
    dataset = client.datasets(name=args.dataset_name, workspace=args.workspace)

    if dataset is None:
        print(f"Error: dataset '{args.dataset_name}' not found.")
        sys.exit(1)

    # Native Argilla exports
    if args.to_hub:
        print(f"Pushing to HuggingFace Hub: {args.to_hub}")
        dataset.to_hub(repo_id=args.to_hub)
        print("Done.")

    if args.to_disk:
        print(f"Saving to disk: {args.to_disk}")
        dataset.to_disk(path=args.to_disk)
        print("Done.")

    # JSONL export for custom pipelines
    if args.to_jsonl:
        print(f"Exporting to JSONL: {args.to_jsonl}")
        records = dataset.records(with_responses=True).to_list(flatten=True)
        with open(args.to_jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"Exported {len(records)} records.")


if __name__ == "__main__":
    main()