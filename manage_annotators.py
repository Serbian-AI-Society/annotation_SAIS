"""
Manage annotator access to the NanoBEIR-sr Argilla workspace.

Annotators sign in at the Space URL using their HuggingFace account.
After they log in for the first time their account is created in Argilla,
but they cannot see any datasets until they are added to the workspace.
Use this script to add or remove them, and to list who currently has access.

Usage:
    # List current annotators
    python manage_annotators.py list

    # Add one or more annotators (use their HuggingFace username)
    python manage_annotators.py add marko_petrovic ana_ivanovic

    # Remove an annotator
    python manage_annotators.py remove marko_petrovic

Annotators must log in at the Space URL at least once before you can add them
— their account only exists in Argilla after their first login.

Space URL: https://serbian-ai-society-argilla-annotation.hf.space

Environment variables (or pass as args):
    ARGILLA_API_URL
    ARGILLA_API_KEY
"""

import argparse
import os
import sys
import warnings

import argilla as rg


def get_client(api_url: str, api_key: str) -> rg.Argilla:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rg.Argilla(api_url=api_url, api_key=api_key)


def cmd_list(client: rg.Argilla, workspace: str) -> None:
    ws = client.workspaces(workspace)

    # All registered users
    all_users = {u.username: u for u in client.users}

    # Users in the workspace
    ws_usernames = {u.username for u in ws.users}

    print(f"\nWorkspace: {workspace}")
    print(f"{'Username':<30}  {'Role':<12}  In workspace")
    print("-" * 60)
    for username, user in sorted(all_users.items()):
        if user.role.value == "owner":
            continue  # skip owner accounts
        in_ws = "yes" if username in ws_usernames else "no — needs to be added"
        print(f"  {username:<28}  {str(user.role.value):<12}  {in_ws}")

    if not any(u.role.value != "owner" for u in all_users.values()):
        print("  (no annotators registered yet — they must log in first)")
    print()


def cmd_add(client: rg.Argilla, workspace: str, usernames: list) -> None:
    ws = client.workspaces(workspace)
    ws_usernames = {u.username for u in ws.users}

    for username in usernames:
        if username in ws_usernames:
            print(f"  {username}: already in workspace, skipping.")
            continue

        user = client.users(username=username)
        if user is None:
            print(
                f"  {username}: not found — they must log in at the Space URL first, "
                f"then run this command again."
            )
            continue

        ws.add_user(user)
        print(f"  {username}: added to workspace '{workspace}'. They can now annotate.")


def cmd_remove(client: rg.Argilla, workspace: str, usernames: list) -> None:
    ws = client.workspaces(workspace)
    ws_usernames = {u.username for u in ws.users}

    for username in usernames:
        if username not in ws_usernames:
            print(f"  {username}: not in workspace, nothing to do.")
            continue

        user = client.users(username=username)
        if user is None:
            print(f"  {username}: user not found.")
            continue

        ws.remove_user(user)
        print(f"  {username}: removed from workspace '{workspace}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage annotator access to the NanoBEIR-sr Argilla workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all registered users and their workspace access")

    p_add = sub.add_parser("add", help="Add annotators to the workspace")
    p_add.add_argument("usernames", nargs="+", metavar="USERNAME",
                       help="HuggingFace username(s) to add")

    p_remove = sub.add_parser("remove", help="Remove annotators from the workspace")
    p_remove.add_argument("usernames", nargs="+", metavar="USERNAME",
                          help="HuggingFace username(s) to remove")

    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set ARGILLA_API_URL / ARGILLA_API_KEY)")
        sys.exit(1)

    client = get_client(args.api_url, args.api_key)

    if args.command == "list":
        cmd_list(client, args.workspace)
    elif args.command == "add":
        cmd_add(client, args.workspace, args.usernames)
    elif args.command == "remove":
        cmd_remove(client, args.workspace, args.usernames)


if __name__ == "__main__":
    main()
