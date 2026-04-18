"""
Create a calibration dataset for measuring inter-annotator agreement (IAA).

Samples a small number of records per benchmark from the main NanoBEIR-sr dataset
and uploads them to a separate calibration dataset where every annotator must
annotate every record. This enables computing inter-annotator agreement (Cohen's
kappa) to verify that annotators are calibrated with each other.

Samples 1 record per benchmark = 13 records total, alternating between
query and passage types so both are represented. Each annotator annotates
all 13 — 15–30 minutes of work.

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

from load_nanobeir import BENCHMARKS, build_settings

BENCHMARK_NAMES = [b["name"] for b in BENCHMARKS]

DEFAULT_CALIBRATION_NAME = "NanoBEIR-sr-calibration"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_records(dataset: rg.Dataset) -> list:
    """
    Sample exactly 1 record per benchmark from the main dataset (13 records total).

    Alternates between query and passage across benchmarks so the calibration
    set has an even mix of both types. Prefers pending records.
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
    for i, bm in enumerate(BENCHMARK_NAMES):
        bm_pool = pool.get(bm, {"query": [], "passage": []})

        # Alternate preferred type: even index → query, odd index → passage
        preferred, fallback = ("query", "passage") if i % 2 == 0 else ("passage", "query")

        if bm_pool[preferred]:
            sampled.append(random.choice(bm_pool[preferred]))
        elif bm_pool[fallback]:
            sampled.append(random.choice(bm_pool[fallback]))
        else:
            print(f"  Warning: {bm} — no pending records available, skipping.")

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
    print(f"\nSampling 1 record per benchmark from '{args.dataset_name}' (13 total)...")
    sampled = sample_records(main_dataset)
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
    print(f"  — {sum(1 for r in cal_records if r.metadata.get('record_type') == 'query')} queries, "
          f"{sum(1 for r in cal_records if r.metadata.get('record_type') == 'passage')} passages")
    print()
    print("Next steps:")
    print(f"  1. Ask every annotator to open '{args.calibration_name}' in Argilla")
    print(f"     and annotate all {len(cal_records)} records.")
    print(f"  2. Each record requires {min_submitted} submissions to be marked complete.")
    print("  3. Once all annotators are done, run:")
    print("       python compute_agreement.py")


if __name__ == "__main__":
    main()
