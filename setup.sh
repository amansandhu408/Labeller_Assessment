#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Environment setup complete."
echo "Activate with: source .venv/bin/activate"
echo "For the full Kaggle experiment, see: notebooks/labeller_assessment.ipynb"
