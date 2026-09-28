#!/usr/bin/env python3
"""Command-line entry point for the Labellerr assessment demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_model import XClipActionRecognizer
from object_model import YOLOEObjectDetector
from parallel_pipeline import run_parallel_video


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AI-assisted egocentric video annotation pipeline."
    )
    parser.add_argument("--video", required=True, help="Path to an egocentric video file.")
    parser.add_argument(
        "--action-weights",
        default="weights/action_head.pt",
        help="Path to the trained X-CLIP action-head checkpoint.",
    )
    parser.add_argument(
        "--object-checkpoint",
        default="yoloe-26s-seg.pt",
        help="YOLOE checkpoint. Ultralytics downloads it on first use if absent.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: outputs/predictions/<video>_predictions.json).",
    )
    parser.add_argument("--frame-fps", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--action-threshold", type=float, default=0.30)
    parser.add_argument("--object-threshold", type=float, default=0.25)
    parser.add_argument("--temporal-tolerance", type=float, default=0.25)
    parser.add_argument("--auto-accept-threshold", type=float, default=0.85)
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    action_weights = Path(args.action_weights)
    if not action_weights.exists():
        raise FileNotFoundError(
            f"Action-head weights not found: {action_weights}. "
            "Run notebooks/labeller_assessment.ipynb first or provide --action-weights."
        )

    action_model = XClipActionRecognizer(action_weights)
    object_model = YOLOEObjectDetector(args.object_checkpoint)

    result = run_parallel_video(
        video,
        action_model=action_model,
        object_model=object_model,
        frame_fps=args.frame_fps,
        batch_size=args.batch_size,
        object_conf=args.object_threshold,
        action_threshold=args.action_threshold,
        temporal_tolerance=args.temporal_tolerance,
        auto_accept_threshold=args.auto_accept_threshold,
    )

    output = (
        Path(args.output)
        if args.output
        else Path("outputs") / "predictions" / f"{video.stem}_predictions.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved predictions to {output}")
    print(f"Action predictions: {len(result['action_predictions'])}")
    print(f"Object detections: {len(result['object_detections'])}")
    print(f"Fused annotations: {len(result['annotations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
