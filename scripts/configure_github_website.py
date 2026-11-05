"""Configure Pages and main-branch protection with an authenticated admin gh CLI.

The default prints a plan. --apply performs the authorized setup, preserving
existing stricter review requirements and the owner's custom-domain settings.
"""
from __future__ import annotations

import argparse
import json
import subprocess

REPOSITORY = "NANInithin/AgentQuantix"
API = f"repos/{REPOSITORY}"
CHECKS = (
    "build + install", "pinned media catalog drift", "windows-latest / py3.10",
    "windows-latest / py3.13", "ubuntu-latest / py3.10", "ubuntu-latest / py3.13",
    "hub >=1.0", "hub ==0.36.*", "lint", "website build",
)


def api(path, method="GET", body=None, missing=False):
    command = ["gh", "api", "--method", method, f"{API}/{path}".rstrip("/")]
    if body is not None:
        command += ["--input", "-"]
    result = subprocess.run(command, input=json.dumps(body) if body is not None else None,
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        if missing and "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(f"{method} {path}: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else None


def enabled(value):
    return value.get("enabled", False) if isinstance(value, dict) else bool(value)


def actors(value):
    return {kind: [actor["slug" if kind in ("teams", "apps") else "login"]
                   for actor in value.get(kind, [])] for kind in ("users", "teams", "apps")}


def protection_payload(current):
    status = current.get("required_status_checks") or {}
    checks = {item["context"]: {"context": item["context"], "app_id": item.get("app_id", -1)}
              for item in status.get("checks", [])}
    for context in status.get("contexts", []):
        checks.setdefault(context, {"context": context, "app_id": -1})
    for context in CHECKS:
        checks[context] = {"context": context, "app_id": 15368}  # GitHub Actions
    previous = current.get("required_pull_request_reviews") or {}
    reviews = {"dismiss_stale_reviews": True,
               "require_code_owner_reviews": previous.get("require_code_owner_reviews", False),
               "required_approving_review_count": previous.get("required_approving_review_count", 0),
               "require_last_push_approval": previous.get("require_last_push_approval", False)}
    for field in ("dismissal_restrictions", "bypass_pull_request_allowances"):
        if previous.get(field):
            reviews[field] = actors(previous[field])
    return {
        "required_status_checks": {"strict": True, "checks": list(checks.values())},
        "enforce_admins": True,
        "required_pull_request_reviews": reviews,
        "restrictions": actors(current["restrictions"]) if current.get("restrictions") else None,
        "required_linear_history": enabled(current.get("required_linear_history")),
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": enabled(current.get("block_creations")),
        "required_conversation_resolution": True,
        "lock_branch": enabled(current.get("lock_branch")),
        "allow_fork_syncing": enabled(current.get("allow_fork_syncing")),
    }


def configure_environment():
    existing = api("environments/github-pages", missing=True) or {}
    body = {"deployment_branch_policy": {
        "protected_branches": False, "custom_branch_policies": True}}
    for rule in existing.get("protection_rules", []):
        if rule["type"] == "wait_timer":
            body["wait_timer"] = rule["wait_timer"]
        if rule["type"] == "required_reviewers":
            body["reviewers"] = [{"type": entry["type"], "id": entry["reviewer"]["id"]}
                                 for entry in rule.get("reviewers", [])]
            body["prevent_self_review"] = rule.get("prevent_self_review", False)
    if "can_admins_bypass" in existing:
        body["can_admins_bypass"] = existing["can_admins_bypass"]
    api("environments/github-pages", "PUT", body)
    policies = api("environments/github-pages/deployment-branch-policies?per_page=100")
    entries = policies["branch_policies"]
    if len(entries) >= 100:
        raise RuntimeError("review the large existing deployment-policy list manually")
    if not any(entry["name"] == "main" and entry.get("type", "branch") == "branch"
               for entry in entries):
        api("environments/github-pages/deployment-branch-policies", "POST",
            {"name": "main", "type": "branch"})
    for entry in entries:
        if entry["name"] != "main" or entry.get("type", "branch") != "branch":
            api(f"environments/github-pages/deployment-branch-policies/{entry['id']}", "DELETE")


def verify():
    protection = api("branches/main/protection")
    status = protection.get("required_status_checks") or {}
    checks = {item["context"]: item.get("app_id") for item in status.get("checks", [])}
    assert all(checks.get(name) == 15368 for name in CHECKS), "required CI checks are missing or unbound"
    assert status.get("strict"), "up-to-date branches are not required"
    assert enabled(protection.get("enforce_admins")), "admins can bypass protection"
    assert protection.get("required_pull_request_reviews") is not None, "pull requests are not required"
    assert enabled(protection.get("required_conversation_resolution")), "review conversations can remain open"
    assert not enabled(protection.get("allow_force_pushes")), "force pushes are allowed"
    assert not enabled(protection.get("allow_deletions")), "main can be deleted"
    pages = api("pages")
    assert pages["build_type"] == "workflow", "Pages is not configured for GitHub Actions"
    environment = api("environments/github-pages")
    assert environment["deployment_branch_policy"] == {
        "protected_branches": False, "custom_branch_policies": True}, "deployment policy differs"
    policies = api("environments/github-pages/deployment-branch-policies?per_page=100")
    assert [(p["name"], p.get("type", "branch")) for p in policies["branch_policies"]] == [
        ("main", "branch")], "only main may deploy"
    print("Verified: main protection, all ten checks, Actions Pages, and main-only deployment.")
    print(f"Custom domain (managed by owner): {pages.get('cname') or 'not set yet'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.apply and not args.verify:
        print(json.dumps({"repository": REPOSITORY, "production_branch": "main",
                          "required_checks": list(CHECKS), "pages_source": "workflow",
                          "deployment_branches": ["main"], "custom_domain": "owner-managed",
                          "review_policy": "preserve existing approval count; require PRs",
                          "force_pushes": False, "branch_deletion": False,
                          "enforce_admins": True}, indent=2))
        return
    try:
        repository = api("")
        if not repository.get("permissions", {}).get("admin"):
            raise RuntimeError("gh must be signed in with repository administration permission")
        if args.apply:
            # Read protection before any change; preserve existing stronger controls.
            current = api("branches/main/protection", missing=True) or {}
            api("branches/main/protection", "PUT", protection_payload(current))
            pages = api("pages", missing=True)
            api("pages", "PATCH" if pages else "POST", {"build_type": "workflow"})
            configure_environment()
        verify()
    except (OSError, RuntimeError, AssertionError, KeyError) as error:
        parser.exit(1, f"GitHub configuration incomplete: {error}\n")


if __name__ == "__main__":
    main()
