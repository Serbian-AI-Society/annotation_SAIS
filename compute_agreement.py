"""
Compute inter-annotator agreement (IAA) from the calibration dataset.

Reads the NanoBEIR-sr-calibration dataset and for every pair of annotators
that both annotated the same record computes:
  - Cohen's kappa  (standard IAA metric; 0 = chance, 1 = perfect)
  - Percentage exact agreement
  - Mean absolute difference in scores

Interpretation of kappa:
  < 0.20  Poor — annotators are not calibrated, results are unreliable
  0.20–0.40  Fair
  0.40–0.60  Moderate — acceptable for pilot annotation
  0.60–0.80  Substantial — good agreement
  > 0.80  Almost perfect

Run this script once all annotators have finished the calibration dataset.

Usage:
    python compute_agreement.py
    python compute_agreement.py --calibration-name NanoBEIR-sr-calibration

Environment variables (or pass as args):
    ARGILLA_API_URL
    ARGILLA_API_KEY
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations

import argilla as rg

DEFAULT_CALIBRATION_NAME = "NanoBEIR-sr-calibration"


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa for two annotators on the same set of records."""
    assert len(labels_a) == len(labels_b), "Label lists must be the same length"
    n = len(labels_a)
    if n == 0:
        return float("nan")

    categories = sorted(set(labels_a) | set(labels_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Confusion matrix
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        conf[cat_idx[a]][cat_idx[b]] += 1

    p_observed = sum(conf[i][i] for i in range(k)) / n

    row_sums = [sum(conf[i]) for i in range(k)]
    col_sums = [sum(conf[r][i] for r in range(k)) for i in range(k)]
    p_expected = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)

    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1.0 - p_expected)


def interpret_kappa(kappa: float) -> str:
    if kappa < 0.20:
        return "Poor"
    if kappa < 0.40:
        return "Fair"
    if kappa < 0.60:
        return "Moderate"
    if kappa < 0.80:
        return "Substantial"
    return "Almost perfect"


def fetch_annotations(client: rg.Argilla, dataset: rg.Dataset) -> dict:
    """
    Returns: {record_id: {username: score_int_or_None}}
    Only includes records with at least 2 submitted responses.
    """
    user_map = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in client.users:
            user_map[str(u.id)] = u.username

    record_annotations: dict = defaultdict(dict)

    query = rg.Query(filter=rg.Filter([("status", "==", "completed")]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rec in dataset.records(query=query, with_responses=True):
            rec_id = rec._model.external_id or str(rec._model.id)
            for resp in (rec._model.responses or []):
                if resp.status.value != "submitted":
                    continue
                uid = str(resp.user_id)
                username = user_map.get(uid, uid[:8])
                score_raw = (resp.values.get("quality_score") or {}).get("value", "")
                score_str = str(score_raw).strip()
                score = int(score_str[0]) if score_str and score_str[0].isdigit() else None
                record_annotations[rec_id][username] = score

    # Keep only records annotated by at least 2 annotators
    return {
        rec_id: ann
        for rec_id, ann in record_annotations.items()
        if len(ann) >= 2
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement from the calibration dataset"
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--calibration-name", default=DEFAULT_CALIBRATION_NAME)
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)

    dataset = client.datasets(name=args.calibration_name, workspace=args.workspace)
    if dataset is None:
        print(f"Error: '{args.calibration_name}' not found.")
        print("Run create_calibration_set.py first.")
        sys.exit(1)

    print(f"Loading annotations from '{args.calibration_name}'...", flush=True)
    annotations = fetch_annotations(client, dataset)

    if not annotations:
        print("No records with 2+ annotations found yet.")
        print("Ask all annotators to complete the calibration dataset first.")
        sys.exit(0)

    # Collect all annotators
    all_annotators = sorted({u for ann in annotations.values() for u in ann})
    print(f"\nAnnotators with overlap: {all_annotators}")
    print(f"Records with 2+ annotations: {len(annotations)}\n")

    # Pairwise agreement
    print("=" * 60)
    print("Pairwise inter-annotator agreement")
    print("=" * 60)

    for ann_a, ann_b in combinations(all_annotators, 2):
        shared_ids = [
            rec_id for rec_id, ann in annotations.items()
            if ann_a in ann and ann_b in ann
            and ann[ann_a] is not None and ann[ann_b] is not None
        ]

        if not shared_ids:
            print(f"\n{ann_a} vs {ann_b}: no shared records")
            continue

        scores_a = [annotations[rid][ann_a] for rid in shared_ids]
        scores_b = [annotations[rid][ann_b] for rid in shared_ids]

        kappa = cohen_kappa(scores_a, scores_b)
        exact = sum(a == b for a, b in zip(scores_a, scores_b)) / len(shared_ids)
        mean_diff = sum(abs(a - b) for a, b in zip(scores_a, scores_b)) / len(shared_ids)

        print(f"\n  {ann_a}  vs  {ann_b}  ({len(shared_ids)} shared records)")
        print(f"    Cohen's kappa:      {kappa:.3f}  ({interpret_kappa(kappa)})")
        print(f"    Exact agreement:    {exact*100:.1f}%")
        print(f"    Mean score diff:    {mean_diff:.2f} points")

    # Per-annotator score distributions on calibration set
    print("\n" + "=" * 60)
    print("Per-annotator score distribution on calibration records")
    print("=" * 60)
    print(f"\n  {'Annotator':<25}  {'N':>4}  {'1':>4}  {'2':>4}  {'3':>4}  {'4':>4}  {'5':>4}  {'Avg':>6}")
    print("  " + "-" * 58)

    annotator_scores: dict = defaultdict(list)
    for ann in annotations.values():
        for username, score in ann.items():
            if score is not None:
                annotator_scores[username].append(score)

    for username in all_annotators:
        scores = annotator_scores.get(username, [])
        if not scores:
            continue
        counts = {i: scores.count(i) for i in range(1, 6)}
        avg = sum(scores) / len(scores)
        print(
            f"  {username:<25}  {len(scores):>4}  "
            + "  ".join(f"{counts[i]:>4}" for i in range(1, 6))
            + f"  {avg:>6.2f}"
        )

    print()


if __name__ == "__main__":
    main()
