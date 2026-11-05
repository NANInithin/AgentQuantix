"""Adding website protection must preserve existing stronger branch controls."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "configure_github_website", Path(__file__).resolve().parents[1]
    / "scripts" / "configure_github_website.py")
settings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(settings)


def test_website_setup_preserves_existing_review_and_push_requirements():
    current = {
        "required_status_checks": {
            "contexts": ["custom security scan"],
            "checks": [{"context": "custom security scan", "app_id": 99}],
        },
        "required_pull_request_reviews": {
            "required_approving_review_count": 2,
            "require_code_owner_reviews": True,
            "require_last_push_approval": True,
            "dismissal_restrictions": {"users": [{"login": "owner"}]},
        },
        "restrictions": {"users": [{"login": "owner"}], "teams": [{"slug": "maintainers"}]},
        "required_linear_history": {"enabled": True},
        "required_signatures": {"enabled": True},
    }
    payload = settings.protection_payload(current)
    reviews = payload["required_pull_request_reviews"]
    assert reviews["required_approving_review_count"] == 2
    assert reviews["require_code_owner_reviews"]
    assert reviews["require_last_push_approval"]
    assert reviews["dismissal_restrictions"]["users"] == ["owner"]
    assert payload["restrictions"]["teams"] == ["maintainers"]
    assert payload["required_linear_history"]
    assert {"context": "custom security scan", "app_id": 99} in (
        payload["required_status_checks"]["checks"])
    # Signature enforcement has its own endpoint and is not changed by setup.
    assert "required_signatures" not in payload


def test_website_setup_closes_bypasses_without_locking_out_solo_reviews():
    payload = settings.protection_payload({})
    assert payload["required_pull_request_reviews"]["required_approving_review_count"] == 0
    assert payload["enforce_admins"]
    assert payload["required_conversation_resolution"]
    assert not payload["allow_force_pushes"]
    assert not payload["allow_deletions"]
    checks = payload["required_status_checks"]["checks"]
    assert {"context": "website build", "app_id": 15368} in checks
    assert {check["context"] for check in checks}.issuperset(settings.CHECKS)
