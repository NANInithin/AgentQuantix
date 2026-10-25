"""Explicit, revision-bound native runtime installation and update management."""
from .spec import RuntimeCancelled, RuntimeDecisionRequired, RuntimeErrorBase, RuntimeHandle, RuntimeSpec, RUNTIMES
from .manager import RuntimeManager, RuntimeSession, current_session, selected_source

__all__ = ["RuntimeCancelled", "RuntimeDecisionRequired", "RuntimeErrorBase", "RuntimeHandle", "RuntimeSpec",
           "RUNTIMES", "RuntimeManager", "RuntimeSession", "current_session", "selected_source"]
