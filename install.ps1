# AgentQuantix bootstrap installer (Windows).
#
#   irm https://raw.githubusercontent.com/<you>/AgentQuantix/main/install.ps1 | iex
#
# or, from a checkout:
#
#   .\install.ps1
#
# Installs uv (a self-contained Python toolchain manager) and then AgentQuantix
# as an isolated tool - no admin rights, no collision with system Python. It
# does NOT install Visual Studio Build Tools, CMake or CUDA; those want winget
# and usually an elevated prompt, so `aqx bootstrap` prints the commands for
# you to read before running them.

$ErrorActionPreference = "Stop"

function Say  { param($m) Write-Host "`n$m" -ForegroundColor White }
function Warn { param($m) Write-Host "  $m" -ForegroundColor Yellow }

$RepoUrl = if ($env:AQX_REPO) { $env:AQX_REPO } else { "https://github.com/NANInithin/AgentQuantix" }
$Ref     = if ($env:AQX_REF)  { $env:AQX_REF }  else { "main" }

Say "AgentQuantix installer"

# ---- uv --------------------------------------------------------------------
# uv rather than pip: it brings its own Python if the system one is too old or
# absent, and `uv tool install` yields an isolated environment with the CLI on
# PATH. A fresh Windows VM often has no Python at all.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # The installer edits the user PATH, which does not reach this process.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Warn "uv is still not on PATH - open a new terminal and re-run."
    exit 1
}
Write-Host "  uv $((uv --version) -split ' ' | Select-Object -Last 1)"

# ---- AgentQuantix ----------------------------------------------------------
# From a local checkout when this script sits next to pyproject.toml, otherwise
# from git. [all] pulls hf_xet (large downloads) and ninja (much faster builds).
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Here -and (Test-Path (Join-Path $Here "pyproject.toml"))) {
    Say "Installing AgentQuantix from $Here"
    # --reinstall rebuilds rather than reusing uv's cached wheel for
    # this path; without it an edited source can install as the old build.
    uv tool install --force --reinstall "$Here[all]"
} else {
    Say "Installing AgentQuantix from $RepoUrl@$Ref"
    uv tool install --force "git+$RepoUrl@$Ref#egg=agentquantix[all]"
}

uv tool update-shell 2>$null | Out-Null
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

if (-not (Get-Command aqx -ErrorAction SilentlyContinue)) {
    Warn "aqx is installed but not on PATH yet."
    Warn "Open a new terminal, or: `$env:Path = `"`$env:USERPROFILE\.local\bin;`$env:Path`""
    exit 0
}

# ---- the rest --------------------------------------------------------------
Say "Checking prerequisites"
aqx bootstrap --check-only

Write-Host @'

Next:
  $env:HF_TOKEN = "hf_..."  your Hugging Face token, with write permission
                            (or run: hf auth login)
  aqx bootstrap             clones and builds llama.cpp (a few minutes)
  aqx research              the top trending models, sized for this machine

`aqx bootstrap` prints the exact commands for anything it cannot install
itself. Nothing above needed admin; those may.
'@
