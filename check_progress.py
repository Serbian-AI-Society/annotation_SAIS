"""
Annotation progress dashboard for the merged NanoBEIR-sr Argilla dataset.

Shows:
  1. Per-benchmark progress (pending / completed / discarded)
  2. Per-annotator summary (total done + quality score distribution)

Usage:
    python check_progress.py

    # Different dataset or workspace
    python check_progress.py --dataset-name NanoBEIR-sr --workspace argilla

Environment variables (or pass as args):
    ARGILLA_API_URL
    ARGILLA_API_KEY
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

import argilla as rg

from load_nanobeir import BENCHMARKS

BENCHMARK_NAMES = [b["name"] for b in BENCHMARKS]


def fetch_data(dataset: rg.Dataset, client: rg.Argilla):
    """
    Single pass over all records. Returns:
      status_counts[benchmark][record_type][status] = int
      annotator_counts[username][score_label] = int   (score_label: "1"-"5" or "discarded")
      error_cat_counts[category] = int
    """
    # Build user_id → username map
    user_map = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in client.users:
            user_map[str(u.id)] = u.username

    status_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    annotator_counts = defaultdict(lambda: defaultdict(int))
    error_cat_counts: dict = defaultdict(int)

    for status_val in ("pending", "completed", "discarded"):
        query = rg.Query(filter=rg.Filter([("status", "==", status_val)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fetch_responses = status_val != "pending"
            for rec in dataset.records(query=query, with_responses=fetch_responses):
                benchmark = (rec.metadata or {}).get("benchmark", "Unknown")
                rec_type = (rec.metadata or {}).get("record_type", "unknown")
                status_counts[benchmark][rec_type][status_val] += 1

                # Per-annotator stats from submitted responses
                if fetch_responses and rec._model.responses:
                    for resp in rec._model.responses:
                        uid = str(resp.user_id)
                        username = user_map.get(uid, uid[:8])

                        if resp.status.value == "discarded":
                            annotator_counts[username]["discarded"] += 1
                        elif resp.status.value == "submitted":
                            qs = resp.values.get("quality_score")
                            if qs:
                                score = str(qs["value"]).strip()[0]
                                if score.isdigit():
                                    annotator_counts[username][score] += 1
                                else:
                                    annotator_counts[username]["?"] += 1
                            cats_raw = (resp.values.get("error_categories") or {}).get("value", []) or []
                            for cat in (cats_raw if isinstance(cats_raw, list) else [cats_raw]):
                                error_cat_counts[cat] += 1

    return status_counts, annotator_counts, error_cat_counts


def print_benchmark_table(status_counts: dict) -> int:
    col_w = 14
    print()
    print(f"{'Benchmark':<25}  {'Queries':>{col_w}}  {'Passages':>{col_w}}  {'Total':>{col_w}}")
    print(f"{'':25}  {'done/total':>{col_w}}  {'done/total':>{col_w}}  {'done/total':>{col_w}}")
    print("-" * (25 + 3 * (col_w + 2) + 2))

    def _stats(bdata, rec_type):
        d = bdata.get(rec_type, {})
        done = d.get("completed", 0) + d.get("discarded", 0)
        total = done + d.get("pending", 0)
        return done, total

    def _fmt(done, total):
        if total == 0:
            return "-"
        pct = done * 100 // total
        return f"{done}/{total} ({pct}%)"

    grand_done = grand_total = 0

    for name in BENCHMARK_NAMES:
        bdata = status_counts.get(name, {})

        q_done, q_total = _stats(bdata, "query")
        p_done, p_total = _stats(bdata, "passage")
        t_done = q_done + p_done
        t_total = q_total + p_total
        grand_done += t_done
        grand_total += t_total

        print(
            f"{name:<25}  {_fmt(q_done, q_total):>{col_w}}  "
            f"{_fmt(p_done, p_total):>{col_w}}  {_fmt(t_done, t_total):>{col_w}}"
        )

    print("-" * (25 + 3 * (col_w + 2) + 2))
    grand_pct = grand_done * 100 // grand_total if grand_total else 0
    total_str = f"{grand_done}/{grand_total} ({grand_pct}%)"
    print(f"{'TOTAL':<25}  {'':>{col_w}}  {'':>{col_w}}  {total_str:>{col_w}}")

    return grand_done


def print_annotator_table(annotator_counts: dict) -> None:
    if not annotator_counts:
        print("\n  No annotations submitted yet.\n")
        return

    print()
    print(f"  {'Annotator':<25}  {'Total':>6}  {'1':>4}  {'2':>4}  {'3':>4}  {'4':>4}  {'5':>4}  {'skip':>5}")
    print("  " + "-" * 65)

    for username, scores in sorted(annotator_counts.items(), key=lambda x: -sum(
        v for k, v in x[1].items() if k != "discarded"
    )):
        total_submitted = sum(v for k, v in scores.items() if k.isdigit())
        total_discarded = scores.get("discarded", 0)
        total = total_submitted + total_discarded

        s1 = scores.get("1", 0)
        s2 = scores.get("2", 0)
        s3 = scores.get("3", 0)
        s4 = scores.get("4", 0)
        s5 = scores.get("5", 0)
        sk = total_discarded

        print(
            f"  {username:<25}  {total:>6}  {s1:>4}  {s2:>4}  {s3:>4}  {s4:>4}  {s5:>4}  {sk:>5}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Annotation progress report for the merged NanoBEIR-sr dataset"
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="NanoBEIR-sr")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)

    dataset = client.datasets(name=args.dataset_name, workspace=args.workspace)
    if dataset is None:
        print(f"Error: dataset '{args.dataset_name}' not found in workspace '{args.workspace}'.")
        sys.exit(1)

    print(f"Fetching progress for '{args.dataset_name}' ...", flush=True)
    status_counts, annotator_counts, error_cat_counts = fetch_data(dataset, client)

    print("\n=== Benchmark progress ===")
    print_benchmark_table(status_counts)

    print("\n=== Per-annotator summary ===")
    print("  Columns 1–5 = quality score chosen | skip = record discarded without annotation")
    print_annotator_table(annotator_counts)

    if error_cat_counts:
        total_submitted = sum(
            sum(v for k, v in counts.items() if k.isdigit())
            for counts in annotator_counts.values()
        )
        print("\n=== Error category frequencies ===")
        print(f"  (across {total_submitted} submitted annotations)\n")
        for cat, count in sorted(error_cat_counts.items(), key=lambda x: -x[1]):
            pct = count * 100 // total_submitted if total_submitted else 0
            bar = "█" * (pct // 5)
            print(f"  {cat:<30}  {count:>5}  ({pct:>3}%)  {bar}")
        print()


if __name__ == "__main__":
    main()
