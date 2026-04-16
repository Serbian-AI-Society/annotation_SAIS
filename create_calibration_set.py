"""
Create a calibration dataset for measuring inter-annotator agreement (IAA).

Samples a small number of records per benchmark from the main NanoBEIR-sr dataset
and uploads them to a separate calibration dataset where every annotator must
annotate every record. This enables computing inter-annotator agreement (Cohen's
kappa) to verify that annotators are calibrated with each other.

Default: 2 records per benchmark (1 query + 1 passage) = 26 records total.
Each annotator annotates all 26 — under an hour of work.

Once all annotators have finished the calibration dataset, run:
    python compute_agreement.py

Usage:
    python create_calibration_set.py

    # More records per benchmark
    python create_calibration_set.py --n-per-benchmark 3

    # Require a specific number of annotators per record (default: auto-detect)
    python create_calibration_set.py --min-submitted 3

Environment variables (or pass as args):
    ARGILLA_API_URL
    ARGILLA_API_KEY
"""

import argparse
import os
import random
import sys
import warnings
from collections import defaultdict

import argilla as rg

from load_nanobeir import BENCHMARK_NAMES, build_settings

DEFAULT_CALIBRATION_NAME = "NanoBEIR-sr-calibration"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_records(dataset: rg.Dataset, n_per_benchmark: int) -> list:
    """
    Sample n_per_benchmark records per benchmark from the main dataset.

    Prefers pending records (not yet annotated by anyone).
    Within each benchmark tries to pick 1 query and 1 passage first,
    then fills remaining slots at random from whatever is available.
    """
    pool: dict = defaultdict(lambda: {"query": [], "passage": []})

    print("  Fetching pending records from main dataset...", flush=True)
    query = rg.Query(filter=rg.Filter([("status", "==", "pending")]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rec in dataset.records(query=query, with_responses=False):
            meta = rec.metadata or {}
            bm = meta.get("benchmark", "Unknown")
            rt = meta.get("record_type", "unknown")
            if bm in BENCHMARK_NAMES and rt in ("query", "passage"):
                pool[bm][rt].append(rec)

    sampled = []
    for bm in BENCHMARK_NAMES:
        bm_pool = pool.get(bm, {"query": [], "passage": []})
        chosen = []

        # 1 query + 1 passage first
        if bm_pool["query"]:
            chosen.append(random.choice(bm_pool["query"]))
        if bm_pool["passage"] and len(chosen) < n_per_benchmark:
            chosen.append(random.choice(bm_pool["passage"]))

        # Fill remaining slots randomly
        remaining = n_per_benchmark - len(chosen)
        if remaining > 0:
            rest = [r for r in bm_pool["query"] + bm_pool["passage"] if r not in chosen]
            chosen.extend(random.sample(rest, min(remaining, len(rest))))

        if len(chosen) < n_per_benchmark:
            print(
                f"  Warning: {bm} — only {len(chosen)} pending records available "
                f"(wanted {n_per_benchmark})"
            )
        sampled.extend(chosen)

    return sampled


def build_calibration_records(sampled: list) -> list:
    """Convert sampled main-dataset records into fresh calibration records."""
    records = []
    skipped = 0
    for rec in sampled:
        fields = rec._model.fields or {}
        source_en = fields.get("source_text_en") or ""
        translation_sr = fields.get("translated_text_sr") or ""

        if not source_en or not translation_sr:
            skipped += 1
            continue

        meta = rec.metadata or {}
        # Prefix with 'cal_' so IDs never collide with main dataset records
        cal_id = f"cal_{rec._model.external_id or str(rec._model.id)}"

        records.append(
            rg.Record(
                id=cal_id,
                fields={
                    "source_text_en": source_en,
                    "translated_text_sr": translation_sr,
                },
                metadata={
                    "task_id": meta.get("task_id", ""),
                    "record_type": meta.get("record_type", "unknown"),
                    "benchmark": meta.get("benchmark", "Unknown"),
                },
            )
        )

    if skipped:
        print(f"  Skipped {skipped} records with missing field content.")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create a calibration dataset for inter-annotator agreement measurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="NanoBEIR-sr",
                        help="Source dataset to sample records from")
    parser.add_argument("--calibration-name", default=DEFAULT_CALIBRATION_NAME,
                        help="Name for the new calibration dataset")
    parser.add_argument(
        "--n-per-benchmark", type=int, default=2,
        help="Records to sample per benchmark (2 → 26 total across 13 benchmarks)",
    )
    parser.add_argument(
        "--min-submitted", type=int, default=None,
        help="Annotations required per record before it is marked complete. "
             "Defaults to the number of annotators currently in the workspace.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    random.seed(args.seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)

    # Auto-detect min_submitted from workspace annotator count
    if args.min_submitted is None:
        ws = client.workspaces(args.workspace)
        annotators = [u for u in ws.users if u.role.value != "owner"]
        min_submitted = max(len(annotators), 2)
        print(
            f"Found {len(annotators)} annotator(s) in workspace "
            f"→ min_submitted={min_submitted} "
            f"(every record needs {min_submitted} annotation(s) to be marked complete)"
        )
    else:
        min_submitted = args.min_submitted
        print(f"min_submitted={min_submitted} (set manually)")

    # Verify source dataset exists
    main_dataset = client.datasets(name=args.dataset_name, workspace=args.workspace)
    if main_dataset is None:
        print(f"Error: source dataset '{args.dataset_name}' not found.")
        sys.exit(1)

    # Abort if calibration dataset already exists
    existing = client.datasets(name=args.calibration_name, workspace=args.workspace)
    if existing is not None:
        print(
            f"Error: '{args.calibration_name}' already exists. "
            "Delete it in the Argilla UI first if you want to recreate it."
        )
        sys.exit(1)

    # Sample records
    print(
        f"\nSampling {args.n_per_benchmark} record(s) per benchmark "
        f"from '{args.dataset_name}'..."
    )
    sampled = sample_records(main_dataset, args.n_per_benchmark)
    print(f"Sampled {len(sampled)} records.")

    cal_records = build_calibration_records(sampled)
    if not cal_records:
        print("Error: no records could be built. Aborting.")
        sys.exit(1)

    # Create calibration dataset with overlap distribution
    print(f"\nCreating '{args.calibration_name}' (min_submitted={min_submitted})...")
    settings = build_settings(
        distribution=rg.OverlapTaskDistribution(min_submitted=min_submitted)
    )
    cal_dataset = rg.Dataset(
        name=args.calibration_name,
        workspace=args.workspace,
        settings=settings,
        client=client,
    )
    cal_dataset.create()

    cal_dataset.records.log(cal_records)

    print(f"\nDone. {len(cal_records)} records uploaded to '{args.calibration_name}'.")
    print()
    print("Next steps:")
    print(f"  1. Ask every annotator to open '{args.calibration_name}' in Argilla")
    print(f"     and annotate all {len(cal_records)} records.")
    print(f"  2. Each record requires {min_submitted} submissions to be marked complete.")
    print("  3. Once all annotators are done, run:")
    print("       python compute_agreement.py")


if __name__ == "__main__":
    main()
