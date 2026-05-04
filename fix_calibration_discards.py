"""
Fix accidentally discarded records in the calibration dataset.

When an annotator discards a calibration record instead of submitting it,
the record disappears from their queue. Deleting the discarded response via
the API restores the record to pending for that annotator, so they can
annotate it properly.

Usage:
    # Preview who has discards (dry run, no changes)
    python fix_calibration_discards.py

    # Actually delete the discarded responses
    python fix_calibration_discards.py --fix

    # Target a different calibration dataset
    python fix_calibration_discards.py --calibration-name NanoBEIR-sr-calibration --fix

Environment variables (or pass as args):
    ARGILLA_API_URL
    ARGILLA_API_KEY
"""

import argparse
import os
import sys
import warnings

import argilla as rg


def main():
    parser = argparse.ArgumentParser(
        description="Restore accidentally discarded calibration records to annotator queues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--calibration-name", default="NanoBEIR-sr-calibration")
    parser.add_argument(
        "--fix", action="store_true",
        help="Actually delete the discarded responses. Without this flag, only a preview is shown.",
    )
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)

    # Build user_id → username map
    user_map = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in client.users:
            user_map[str(u.id)] = u.username

    dataset = client.datasets(name=args.calibration_name, workspace=args.workspace)
    if dataset is None:
        print(f"Error: dataset '{args.calibration_name}' not found.")
        sys.exit(1)

    print(f"Scanning '{args.calibration_name}' for discarded responses...\n")

    discards = []  # list of (record_id, response_id, username)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rec in dataset.records(with_responses=True):
            rec_id = rec._model.external_id or str(rec._model.id)
            for resp in (rec._model.responses or []):
                if resp.status.value == "discarded":
                    username = user_map.get(str(resp.user_id), str(resp.user_id)[:8])
                    discards.append((rec_id, str(resp.id), username))

    if not discards:
        print("No discarded responses found. All annotators are on track.")
        return

    print(f"Found {len(discards)} discarded response(s):\n")
    for rec_id, resp_id, username in discards:
        print(f"  annotator={username}  record={rec_id}  response_id={resp_id}")

    if not args.fix:
        print(
            f"\nDry run — no changes made. "
            f"Run with --fix to restore these {len(discards)} record(s) to the annotators' queues."
        )
        return

    print(f"\nDeleting {len(discards)} discarded response(s)...")
    fixed = 0
    failed = 0
    for rec_id, resp_id, username in discards:
        try:
            response = client.http_client.delete(f"/api/v1/responses/{resp_id}")
            response.raise_for_status()
            print(f"  ✓ Restored: annotator={username}  record={rec_id}")
            fixed += 1
        except Exception as exc:
            print(f"  ✗ Failed:   annotator={username}  record={rec_id}  error={exc}")
            failed += 1

    print(f"\nDone. {fixed} restored, {failed} failed.")
    if fixed:
        print(
            "Affected annotators will now see the restored records in their queue again."
        )


if __name__ == "__main__":
    main()
