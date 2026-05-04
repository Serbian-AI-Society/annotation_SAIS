"""
Compute inter-annotator agreement (IAA) from the calibration dataset.

Reads the NanoBEIR-sr-calibration dataset and for every pair of annotators
that both annotated the same record computes:
  - Quadratic weighted Cohen's kappa  (standard for ordinal scales; 0 = chance, 1 = perfect)
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


def cohen_kappa_weighted(labels_a: list, labels_b: list) -> float:
    """
    Quadratic weighted Cohen's kappa for two annotators on the same set of records.

    Quadratic weighting is standard for ordinal scales: a disagreement of 2 points
    is penalised 4× more than a disagreement of 1 point. Unweighted kappa treats
    all disagreements equally, which is wrong for a 1–5 quality score.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        return float("nan")

    # Use the fixed 1–5 ordinal scale so the weight matrix is stable regardless
    # of which scores happen to appear in this particular pair's data.
    categories = list(range(1, 6))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Quadratic weight matrix: w[i][j] = 1 - ((i-j)/(k-1))^2
    weights = [
        [1.0 - ((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)]
        for i in range(k)
    ]

    # Observed frequency matrix
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        if a in cat_idx and b in cat_idx:
            obs[cat_idx[a]][cat_idx[b]] += 1

    row_sums = [sum(obs[i]) for i in range(k)]
    col_sums = [sum(obs[r][i] for r in range(k)) for i in range(k)]

    # Expected frequency matrix under marginal independence
    exp = [[row_sums[i] * col_sums[j] / n for j in range(k)] for i in range(k)]

    numerator = sum(weights[i][j] * obs[i][j] for i in range(k) for j in range(k))
    denominator = sum(weights[i][j] * exp[i][j] for i in range(k) for j in range(k))

    if denominator == 0:
        return 1.0
    return 1.0 - (n - numerator) / (n - denominator)


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
    Returns: {record_id: {username: {"score": int|None, "error_cats": set}}}
    Only includes records with at least 2 submitted responses.
    """
    user_map = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in client.users:
            user_map[str(u.id)] = u.username

    record_annotations: dict = defaultdict(dict)

    # Fetch all records — calibration records stay "pending" indefinitely
    # (min_submitted=100) so filtering by status=="completed" would return nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rec in dataset.records(with_responses=True):
            rec_id = rec._model.external_id or str(rec._model.id)
            for resp in (rec._model.responses or []):
                if resp.status.value != "submitted":
                    continue
                uid = str(resp.user_id)
                username = user_map.get(uid, uid[:8])

                score_raw = (resp.values.get("quality_score") or {}).get("value", "")
                score_str = str(score_raw).strip()
                score = int(score_str[0]) if score_str and score_str[0].isdigit() else None

                cats_raw = (resp.values.get("error_categories") or {}).get("value", []) or []
                error_cats = set(cats_raw) if isinstance(cats_raw, list) else {cats_raw}

                record_annotations[rec_id][username] = {
                    "score": score,
                    "error_cats": error_cats,
                }

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
            and ann[ann_a]["score"] is not None and ann[ann_b]["score"] is not None
        ]

        if not shared_ids:
            print(f"\n{ann_a} vs {ann_b}: no shared records")
            continue

        scores_a = [annotations[rid][ann_a]["score"] for rid in shared_ids]
        scores_b = [annotations[rid][ann_b]["score"] for rid in shared_ids]

        kappa = cohen_kappa_weighted(scores_a, scores_b)
        exact = sum(a == b for a, b in zip(scores_a, scores_b)) / len(shared_ids)
        mean_diff = sum(abs(a - b) for a, b in zip(scores_a, scores_b)) / len(shared_ids)

        print(f"\n  {ann_a}  vs  {ann_b}  ({len(shared_ids)} shared records)")
        print(f"    Weighted kappa:     {kappa:.3f}  ({interpret_kappa(kappa)})")
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
        for username, data in ann.items():
            if data["score"] is not None:
                annotator_scores[username].append(data["score"])

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

    # Error category agreement
    all_cats = sorted({
        cat
        for ann in annotations.values()
        for data in ann.values()
        for cat in data["error_cats"]
    })

    if all_cats:
        print("\n" + "=" * 60)
        print("Error category pairwise agreement (% records where both annotators agree)")
        print("(agree = both selected it, or both did not select it)")
        print("=" * 60)

        for ann_a, ann_b in combinations(all_annotators, 2):
            shared_ids = [
                rec_id for rec_id, ann in annotations.items()
                if ann_a in ann and ann_b in ann
            ]
            if not shared_ids:
                continue

            print(f"\n  {ann_a}  vs  {ann_b}  ({len(shared_ids)} shared records)")
            for cat in all_cats:
                agreed = sum(
                    (cat in annotations[rid][ann_a]["error_cats"]) ==
                    (cat in annotations[rid][ann_b]["error_cats"])
                    for rid in shared_ids
                )
                pct = agreed * 100 / len(shared_ids)
                print(f"    {cat:<30}  {pct:.0f}% agreement")

    print()


if __name__ == "__main__":
    main()
