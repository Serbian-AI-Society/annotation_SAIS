"""
Annotation analytics and progress tracking.

Reports:
- Total records vs completed
- Completed tasks per annotator
- Quality score distribution (overall and per annotator)
- Modification rate (how often annotators correct vs accept)

Usage:
    python analytics.py --api-url <URL> --api-key <KEY>
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import argilla as rg


def run_analytics(api_url: str, api_key: str, workspace: str = "argilla",
                  dataset_name: str = "translation-annotation-sr"):
    """Compute and print annotation analytics."""

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
        pass

    # Collect stats
    total_records = 0
    records_with_responses = 0
    quality_counts = Counter()
    per_annotator_count = Counter()
    per_annotator_quality = defaultdict(Counter)
    modification_counts = {"modified": 0, "accepted": 0}

    no_correction_markers = ("bez ispravki", "no corrections", "")

    for record in dataset.records(with_responses=True):
        total_records += 1

        if not record.responses:
            continue

        has_submitted = False
        for response in record.responses:
            if response.status != "submitted":
                continue

            has_submitted = True

            # Resolve annotator
            annotator = "unknown"
            if hasattr(response, "user_id") and response.user_id:
                uid = str(response.user_id)
                annotator = users_by_id.get(uid, uid)

            per_annotator_count[annotator] += 1

            # Extract values
            quality = None
            corrected = None
            if hasattr(response, "values") and response.values:
                quality = response.values.get("quality_score")
                corrected = response.values.get("corrected_text_sr")

            if isinstance(quality, dict):
                quality = quality.get("value", quality)
            if isinstance(corrected, dict):
                corrected = corrected.get("value", corrected)

            if quality:
                quality_counts[quality] += 1
                per_annotator_quality[annotator][quality] += 1

            # Check modification
            corrected_str = (corrected or "").strip().lower()
            if corrected_str in no_correction_markers:
                modification_counts["accepted"] += 1
            else:
                modification_counts["modified"] += 1

        if has_submitted:
            records_with_responses += 1

    # Print report
    print("=" * 60)
    print(f"  ANNOTATION PROGRESS REPORT: {dataset_name}")
    print("=" * 60)
    print()

    print(f"Total records:     {total_records}")
    print(f"Completed records: {records_with_responses}")
    if total_records > 0:
        pct = (records_with_responses / total_records) * 100
        print(f"Completion rate:   {pct:.1f}%")
    print()

    print("-" * 40)
    print("  TASKS PER ANNOTATOR")
    print("-" * 40)
    if per_annotator_count:
        for annotator, count in per_annotator_count.most_common():
            print(f"  {annotator}: {count}")
    else:
        print("  No submitted annotations yet.")
    print()

    print("-" * 40)
    print("  QUALITY SCORE DISTRIBUTION")
    print("-" * 40)
    total_scored = sum(quality_counts.values())
    if total_scored > 0:
        for label in ("high", "medium", "low"):
            c = quality_counts.get(label, 0)
            pct = (c / total_scored) * 100
            bar = "#" * int(pct / 2)
            print(f"  {label:8s}: {c:4d} ({pct:5.1f}%) {bar}")
    else:
        print("  No quality scores recorded yet.")
    print()

    print("-" * 40)
    print("  MODIFICATION RATE")
    print("-" * 40)
    total_mod = modification_counts["modified"] + modification_counts["accepted"]
    if total_mod > 0:
        mod_pct = (modification_counts["modified"] / total_mod) * 100
        print(f"  Modified:  {modification_counts['modified']} ({mod_pct:.1f}%)")
        print(f"  Accepted:  {modification_counts['accepted']} ({100 - mod_pct:.1f}%)")
    else:
        print("  No data yet.")
    print()

    # Per-annotator quality breakdown
    if per_annotator_quality:
        print("-" * 40)
        print("  QUALITY SCORES BY ANNOTATOR")
        print("-" * 40)
        for annotator in sorted(per_annotator_quality.keys()):
            counts = per_annotator_quality[annotator]
            total_a = sum(counts.values())
            parts = []
            for label in ("high", "medium", "low"):
                c = counts.get(label, 0)
                if c > 0:
                    parts.append(f"{label}={c}")
            print(f"  {annotator}: {total_a} total ({', '.join(parts)})")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Annotation analytics")
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="translation-annotation-sr")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required")
        sys.exit(1)

    run_analytics(args.api_url, args.api_key, args.workspace, args.dataset_name)


if __name__ == "__main__":
    main()
