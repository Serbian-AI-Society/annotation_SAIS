"""
Export completed annotations from Argilla as JSONL.

Each output line matches the product requirements:
{
    "id": "...",
    "corrected_text_sr": "...",
    "quality_score": "low|medium|high",
    "annotator_id": "...",
    "timestamp": "2026-04-02T12:00:00Z",
    "edit_metadata": {
        "was_modified": true,
        "edit_distance": 12
    }
}

Usage:
    python export_annotations.py --output annotations.jsonl
    python export_annotations.py --output annotations.jsonl --status submitted
"""

import argparse
import json
import os
import sys
from datetime import datetime

import argilla as rg


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def export_annotations(api_url: str, api_key: str, output_path: str,
                       workspace: str = "argilla",
                       dataset_name: str = "translation-annotation-sr",
                       status_filter: str = "submitted"):
    """Export annotations from Argilla to JSONL."""

    client = rg.Argilla(api_url=api_url, api_key=api_key)
    dataset = client.datasets(name=dataset_name, workspace=workspace)

    if dataset is None:
        print(f"Error: dataset '{dataset_name}' not found.")
        sys.exit(1)

    # Build user ID -> username mapping
    users_by_id = {}
    try:
        for user in client.users:
            users_by_id[str(user.id)] = user.username
    except Exception:
        pass  # If we can't list users, we'll use raw IDs

    # Fetch records with responses
    if status_filter:
        query = rg.Query(
            filter=rg.Filter([("response.status", "==", status_filter)])
        )
        records = dataset.records(query=query, with_responses=True)
    else:
        records = dataset.records(with_responses=True)

    exported_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            # Get the original translated text for edit distance calculation
            original_sr = record.fields.get("translated_text_sr", "")
            record_id = record.metadata.get("task_id", str(record.id)) if record.metadata else str(record.id)

            # Each record can have multiple responses (from different annotators)
            if not record.responses:
                continue

            for response in record.responses:
                if status_filter and response.status != status_filter:
                    continue

                # Extract response values
                quality = None
                corrected = None

                if hasattr(response, "values") and response.values:
                    quality = response.values.get("quality_score")
                    corrected = response.values.get("corrected_text_sr")

                # Handle different response value formats
                if isinstance(quality, dict):
                    quality = quality.get("value", quality)
                if isinstance(corrected, dict):
                    corrected = corrected.get("value", corrected)

                # Skip incomplete annotations
                if quality is None:
                    continue

                # Determine if text was modified
                corrected_str = corrected or ""
                no_correction_markers = ("bez ispravki", "no corrections", "")
                was_modified = corrected_str.strip().lower() not in no_correction_markers

                # Compute edit distance
                edit_dist = 0
                if was_modified and corrected_str.strip():
                    edit_dist = levenshtein_distance(original_sr, corrected_str)

                # Resolve annotator
                annotator_id = "unknown"
                if hasattr(response, "user_id") and response.user_id:
                    uid = str(response.user_id)
                    annotator_id = users_by_id.get(uid, uid)

                # Build output record
                output = {
                    "id": record_id,
                    "corrected_text_sr": corrected_str if was_modified else "Bez ispravki",
                    "quality_score": quality,
                    "annotator_id": annotator_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "edit_metadata": {
                        "was_modified": was_modified,
                        "edit_distance": edit_dist,
                    },
                }

                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                exported_count += 1

    print(f"Exported {exported_count} annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export annotations from Argilla as JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="translation-annotation-sr")
    parser.add_argument("--status", default="submitted",
                        help="Filter by response status (submitted/draft/all)")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required")
        sys.exit(1)

    status = args.status if args.status != "all" else None
    export_annotations(args.api_url, args.api_key, args.output,
                       args.workspace, args.dataset_name, status)


if __name__ == "__main__":
    main()
