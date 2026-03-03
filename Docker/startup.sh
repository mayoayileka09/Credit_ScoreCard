# #!/bin/sh
# PROJECT_ROOT="/workspaces/ml_project_credit_scorecard"   
# VENV_PATH="$PROJECT_ROOT/.venv"         # Note: env kept in the repo folder

# # Install uv globally first
# curl -LsSf https://astral.sh/uv/install.sh | sh


# make venv

# # Ensure every new shell auto-activates the venv
# echo 'source /workspaces/MyProject/.venv/bin/activate' >> /home/vscode/.bashrc
# echo 'source /workspaces/MyProject/.venv/bin/activate' >> /home/vscode/.zshrc

# echo 'export PYTHONPATH="/workspaces/MyProject/llm_utils:${PYTHONPATH}"' >> ~/.profile

# # Prepare MLFlow
# export MLFLOW_TRACKING_URI="https://.com"
# #export MLFLOW_ARTIFACT_URI = "s3://your-s3-bucket/path"

#!/usr/bin/env bash
set -e

PROJECT_ROOT="/workspaces/ml_project_credit_scorecard"
cd "$PROJECT_ROOT"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

make venv