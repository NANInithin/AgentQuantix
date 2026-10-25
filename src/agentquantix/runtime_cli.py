"""Native runtime maintenance commands backed by the shared tool actions."""
from __future__ import annotations

import json
import sys

from .agent import runtime_tools
from .runtimes import RUNTIMES, RuntimeErrorBase


def _print(value):
    print(json.dumps(value, indent=2, default=str))


def _approve(args, message):
    if args.yes:
        return True
    if not sys.stdin.isatty():
        raise ValueError("explicit --yes is required for noninteractive runtime changes")
    return input(message + " [y/N] ").strip().lower() in ("y", "yes")


def command(args):
    try:
        action = args.runtime_command
        if action in ("list", "show"):
            _print(runtime_tools.call("get_runtime_status", {"engine": getattr(args, "engine", None)}))
            return 0
        if action == "prune":
            preview = runtime_tools.call("prune_runtime", {"engine": args.engine})
            _print(preview)
            if not args.apply:
                return 0
            if not _approve(args, "Remove these obsolete managed runtime builds?"):
                return 0
            _print(runtime_tools.call("prune_runtime", {"engine": args.engine,
                                                        "dry_run": False, "user_approved": True}))
            return 0
        if action in ("adopt", "rollback"):
            details = {"engine": args.engine, "action": action}
            if action == "adopt":
                details["source"] = args.source
            _print(details)
            if not _approve(args, "Verify and register this existing runtime?" if action == "adopt"
                            else "Restore the previous verified runtime installation?"):
                return 0
            details["user_approved"] = True
            _print(runtime_tools.call(action + "_runtime", details))
            return 0
        check = runtime_tools.call("check_runtime_updates", {"engine": args.engine})
        _print(check)
        if action == "check":
            return 0 if check.get("available", check.get("status") != "unavailable") else 1

        # The target displayed here is the one passed unchanged to preparation.
        # The manager resolves/fetches exact SHAs rather than a moving branch.
        target = check.get("target")
        if not target:
            status = runtime_tools.call("get_runtime_status", {"engine": args.engine})
            if action == "install" and status.get("installed"):
                _print({"status": "ready", "runtime": status["runtime"], "update_check": "unavailable"})
                return 0
            print("runtime: no verified upstream target is available; installed runtime was kept")
            return 1
        if args.persistent is None:
            status = runtime_tools.call("get_runtime_status", {"engine": args.engine})
            runtime = status.get("runtime") or {}
            if runtime.get("storage_policy") in ("persistent", "temporary"):
                persistent = runtime["storage_policy"] == "persistent"
            else:
                if not sys.stdin.isatty():
                    print("runtime: select --persistent or --temporary for noninteractive setup")
                    return 2
                answer = input(f"Keep {args.engine} installed for regular use? [Y/n] ").strip().lower()
                persistent = answer not in ("n", "no")
        else:
            persistent = args.persistent
        if not args.yes:
            if not sys.stdin.isatty():
                print("runtime: explicit --yes is required for noninteractive runtime setup")
                return 2
            storage = "persistent" if persistent else "temporary"
            if input(f"Build and validate this {storage} runtime target? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Runtime setup cancelled.")
                return 0
        result = runtime_tools.call("prepare_runtime", {
            "engine": args.engine, "persistent": persistent, "user_approved": True,
            "update_approved": action == "update", "target_revision": target,
            "compute_backend": args.compute_backend, "targets": args.target,
        })
        _print(result)
        return 0 if result.get("status") == "ready" else 2
    except (RuntimeErrorBase, ValueError, OSError) as error:
        print(f"runtime: {error}")
        return 1


def add_parser(subparsers):
    parser = subparsers.add_parser("runtime", help="inspect, install and update native engines")
    commands = parser.add_subparsers(dest="runtime_command", required=True)
    listing = commands.add_parser("list", help="show installed runtime revisions and storage")
    listing.set_defaults(func=command)
    for name in ("show", "check", "install", "update", "adopt", "rollback", "prune"):
        item = commands.add_parser(name)
        item.add_argument("engine", choices=list(RUNTIMES))
        if name in ("install", "update"):
            storage = item.add_mutually_exclusive_group()
            storage.add_argument("--persistent", dest="persistent", action="store_true")
            storage.add_argument("--temporary", dest="persistent", action="store_false")
            item.set_defaults(persistent=None)
            item.add_argument("--compute-backend", default="auto")
            item.add_argument("--target", action="append", default=[], help="additional native executable target")
            item.add_argument("--yes", "-y", action="store_true", help="approve this runtime setup/update; independent of model --yes")
        elif name in ("adopt", "rollback", "prune"):
            item.add_argument("--yes", "-y", action="store_true", help="approve this runtime maintenance action")
            if name == "adopt":
                item.add_argument("source", help="existing original runtime checkout to verify and adopt")
            if name == "prune":
                item.add_argument("--apply", action="store_true", help="remove listed obsolete builds; default is inventory only")
        item.set_defaults(func=command)
