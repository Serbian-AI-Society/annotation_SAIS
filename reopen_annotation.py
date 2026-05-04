"""
Re-open a completed main-dataset record for re-annotation.

If a submitted annotation is wrong or low quality, deleting the response
returns the record to pending so another annotator can pick it up.

Usage:
    # Preview — show all submissions for a record
    python reopen_annotation.py --record-id NanoArguAna_query_42

    # Delete a specific annotator's response (re-opens the record for them)
    python reopen_annotation.py --record-id NanoArguAna_query_42 --annotator marko_petrovic --fix

    # Delete ALL responses for a record (re-opens it for everyone)
    python reopen_annotation.py --record-id NanoArguAna_query_42 --all --fix

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
        description="Re-open a completed record for re-annotation by deleting its response(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="NanoBEIR-sr")
    parser.add_argument("--record-id", required=True,
                        help="External record ID, e.g. NanoArguAna_query_42")
    parser.add_argument("--annotator", default=None,
                        help="Delete only this annotator's response (HuggingFace username)")
    parser.add_argument("--all", dest="all_responses", action="store_true",
                        help="Delete ALL responses for this record (re-opens for everyone)")
    parser.add_argument("--fix", action="store_true",
                        help="Actually delete. Without this flag only a preview is shown.")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    if not args.annotator and not args.all_responses:
        print("Error: specify --annotator <username> or --all")
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)

    user_map = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in client.users:
            user_map[str(u.id)] = u.username
    username_to_id = {v: k for k, v in user_map.items()}

    dataset = client.datasets(name=args.dataset_name, workspace=args.workspace)
    if dataset is None:
        print(f"Error: dataset '{args.dataset_name}' not found.")
        sys.exit(1)

    # Find the record by external_id
    target_rec = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rec in dataset.records(with_responses=True):
            if (rec._model.external_id == args.record_id
                    or str(rec._model.id) == args.record_id):
                target_rec = rec
                break

    if target_rec is None:
        print(f"Error: record '{args.record_id}' not found in '{args.dataset_name}'.")
        sys.exit(1)

    responses = target_rec._model.responses or []
    if not responses:
        print(f"Record '{args.record_id}' has no responses — already pending.")
        return

    print(f"\nRecord: {args.record_id}  (status: {target_rec.status})")
    print(f"Responses:\n")
    for resp in responses:
        username = user_map.get(str(resp.user_id), str(resp.user_id)[:8])
        score_raw = (resp.values or {}).get("quality_score", {})
        score = (score_raw.get("value", "?") if isinstance(score_raw, dict) else "?")
        print(f"  annotator={username}  status={resp.status.value}  score={score}  id={resp.id}")

    # Select responses to delete
    to_delete = []
    if args.all_responses:
        to_delete = [(str(r.id), user_map.get(str(r.user_id), str(r.user_id)[:8]))
                     for r in responses]
    else:
        for resp in responses:
            username = user_map.get(str(resp.user_id), str(resp.user_id)[:8])
            if username == args.annotator or str(resp.user_id) == username_to_id.get(args.annotator):
                to_delete.append((str(resp.id), username))
        if not to_delete:
            print(f"\nNo response found for annotator '{args.annotator}'.")
            return

    if not args.fix:
        print(f"\nDry run — would delete {len(to_delete)} response(s):")
        for resp_id, username in to_delete:
            print(f"  {username}  response_id={resp_id}")
        print("\nRun with --fix to apply.")
        return

    answer = input(f"\nDelete {len(to_delete)} response(s) for record '{args.record_id}'? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        return

    for resp_id, username in to_delete:
        try:
            r = client.http_client.delete(f"/api/v1/responses/{resp_id}")
            r.raise_for_status()
            print(f"  ✓ Deleted response for {username} — record is pending again.")
        except Exception as exc:
            print(f"  ✗ Failed for {username}: {exc}")


if __name__ == "__main__":
    main()
