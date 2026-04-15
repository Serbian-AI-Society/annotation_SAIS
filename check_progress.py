"""
Annotation progress dashboard for the merged NanoBEIR-sr Argilla dataset.

Shows per-benchmark and per-record-type breakdown of annotation progress
(pending / submitted / discarded), which the standard Argilla UI does not
provide for a merged single-dataset setup.

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

# Ordered benchmark list — determines row order in the report.
BENCHMARK_NAMES = [
    "NanoArguAna",
    "NanoTouche2020",
    "NanoSciFact",
    "NanoSCIDOCS",
    "NanoNQ",
    "NanoNFCorpus",
    "NanoMSMARCO",
    "NanoFiQA2018",
    "NanoHotpotQA",
    "NanoFEVER",
    "NanoDBPedia",
    "NanoQuoraRetrieval",
    "NanoClimateFEVER",
]


def fetch_status_counts(dataset: rg.Dataset) -> dict:
    """
    Fetch all records and group them by (benchmark, record_type, status).

    Returns:
        counts[benchmark][record_type][status] = int
        where status is one of: "pending", "completed", "discarded"
    """
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Fetch all records with their response info.
    # With 2,811 records this is fast (~5 s); avoids 13 separate API calls.
    for status_val in ("pending", "completed", "discarded"):
        query = rg.Query(filter=rg.Filter([("status", "==", status_val)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rec in dataset.records(query=query, with_responses=False):
                benchmark = (rec.metadata or {}).get("benchmark", "Unknown")
                rec_type = (rec.metadata or {}).get("record_type", "unknown")
                counts[benchmark][rec_type][status_val] += 1

    return counts


def print_report(counts: dict) -> None:
    """Print the per-benchmark progress table."""
    col_w = 14

    # Header
    print()
    print(f"{'Benchmark':<25}  {'Queries':>{col_w}}  {'Passages':>{col_w}}  {'Total':>{col_w}}")
    print(f"{'':25}  {'done/total':>{col_w}}  {'done/total':>{col_w}}  {'done/total':>{col_w}}")
    print("-" * (25 + 3 * (col_w + 2) + 2))

    grand_done = grand_total = 0

    for name in BENCHMARK_NAMES:
        bdata = counts.get(name, {})

        def _stats(rec_type: str):
            d = bdata.get(rec_type, {})
            done = d.get("completed", 0) + d.get("discarded", 0)
            total = done + d.get("pending", 0)
            return done, total

        q_done, q_total = _stats("query")
        p_done, p_total = _stats("passage")
        t_done = q_done + p_done
        t_total = q_total + p_total

        grand_done += t_done
        grand_total += t_total

        def _fmt(done, total):
            if total == 0:
                return "-"
            pct = done * 100 // total
            return f"{done}/{total} ({pct}%)"

        print(
            f"{name:<25}  {_fmt(q_done, q_total):>{col_w}}  "
            f"{_fmt(p_done, p_total):>{col_w}}  {_fmt(t_done, t_total):>{col_w}}"
        )

    print("-" * (25 + 3 * (col_w + 2) + 2))
    grand_pct = grand_done * 100 // grand_total if grand_total else 0
    total_str = f"{grand_done}/{grand_total} ({grand_pct}%)"
    print(f"{'TOTAL':<25}  {'':>{col_w}}  {'':>{col_w}}  {total_str:>{col_w}}")
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
    counts = fetch_status_counts(dataset)
    print_report(counts)


if __name__ == "__main__":
    main()
