#!/usr/bin/env bash
# AgentQuantix bootstrap installer (Linux / macOS).
#
#   curl -fsSL https://raw.githubusercontent.com/<you>/AgentQuantix/main/install.sh | bash
#
# or, from a checkout:
#
#   ./install.sh
#
# What this does and does NOT do matters. It installs uv (a self-contained
# Python toolchain manager) and then AgentQuantix as an isolated tool, which
# needs no root and cannot collide with system Python. It does NOT install
# system packages - git, cmake, a compiler and optionally CUDA still need your
# package manager and usually root, so `aqx bootstrap` prints those commands
# for you to read before running them.
set -euo pipefail

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '  \033[33m%s\033[0m\n' "$*"; }

REPO_URL="${AQX_REPO:-https://github.com/NANInithin/AgentQuantix}"
REF="${AQX_REF:-main}"

say "AgentQuantix installer"

# ---- uv -------------------------------------------------------------------
# uv rather than pip: it installs its own Python if the system one is too old,
# and `uv tool install` gives an isolated environment with the CLI on PATH.
# A fresh VM frequently has no pip, or a python3 that is 3.8.
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv here and edits the shell profile, but that does not
    # affect the already-running shell.
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { warn "uv is still not on PATH - open a new shell and re-run."; exit 1; }
echo "  uv $(uv --version | awk '{print $2}')"

# ---- AgentQuantix ---------------------------------------------------------
# Installed from a local checkout when this script sits next to pyproject.toml,
# otherwise from git. [all] pulls hf_xet (large downloads) and ninja (fast
# builds), both of which materially change what the tool can do.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Git Bash / MSYS hands out POSIX paths (/c/Users/...), but uv is a native
# Windows binary and reads that as a relative path under C:\ - producing
# "Distribution not found at file:///C:/c/Users/...". cygpath -m gives the
# mixed form (C:/Users/...) that both understand. A no-op everywhere else.
if command -v cygpath >/dev/null 2>&1; then
    HERE="$(cygpath -m "$HERE")"
fi

if [ -f "$HERE/pyproject.toml" ]; then
    say "Installing AgentQuantix from $HERE"
    # --reinstall rebuilds rather than reusing uv's cached wheel for this
    # path. Without it, editing the source and reinstalling silently keeps
    # the previous build.
    uv tool install --force --reinstall "${HERE}[all]"
else
    say "Installing AgentQuantix from $REPO_URL@$REF"
    uv tool install --force "git+$REPO_URL@$REF#egg=agentquantix[all]"
fi

uv tool update-shell >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"

command -v aqx >/dev/null 2>&1 || {
    warn "aqx is installed but not on PATH yet."
    warn "Open a new shell, or: export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 0
}

# ---- the rest -------------------------------------------------------------
say "Checking prerequisites"
aqx bootstrap --check-only || true

cat <<'EOF'

Next:
  export MLE2=hf_...        your Hugging Face token, with write permission
  aqx bootstrap             clones and builds llama.cpp (a few minutes)
  aqx research              the top trending models, sized for this machine

`aqx bootstrap` prints the exact commands for anything it cannot install
itself. Nothing above needed root; those may.
EOF
