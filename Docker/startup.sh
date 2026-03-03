#!/usr/bin/env bash
set -e

PROJECT_ROOT="/workspaces/ml_project_credit_scorecard"
VENV_DIR="/home/vscode/venvs/credit_scorecard"

cd "$PROJECT_ROOT"

# Install uv (optional, but keep if you want it)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Create venv OUTSIDE /workspaces
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"

# Activate and install deps
source "$VENV_DIR/bin/activate"

# Make uv safer on mounted filesystems (avoids hardlink/temp issues)
export UV_LINK_MODE=copy

# Install requirements (prefer uv if you want; pip fallback is fine too)
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  python -m pip install -U pip
  python -m pip install -r requirements.txt
fi

echo "✅ Using venv at $VENV_DIR"