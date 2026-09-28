#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

source .venv/bin/activate

if [ "$#" -eq 0 ]; then
  echo "Running repository preflight..."
  python -m compileall -q src
  python src/run_demo.py --help >/dev/null
  echo "Preflight passed."
  echo
  echo "To run inference on a local video:"
  echo "  ./run.sh path/to/video.mp4"
  echo
  echo "The trained action-head checkpoint is expected at:"
  echo "  weights/action_head.pt"
  exit 0
fi

VIDEO_PATH="$1"
ACTION_WEIGHTS="${2:-weights/action_head.pt}"

python src/run_demo.py \
  --video "$VIDEO_PATH" \
  --action-weights "$ACTION_WEIGHTS"
