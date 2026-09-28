"""Parallel action + object annotation pipeline for the assessment.

The video is decoded once into a shared timestamped frame stream. Independent
workers then run X-CLIP action recognition and YOLOE object detection over the
same frame data. Their outputs are temporally aligned and routed by confidence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from decord import VideoReader, cpu


DEFAULT_FRAME_FPS = 4.0
DEFAULT_ACTION_WINDOW_SECONDS = 2.0
DEFAULT_ACTION_STRIDE_SECONDS = 1.0
DEFAULT_NUM_FRAMES = 8
DEFAULT_ACTION_THRESHOLD = 0.30
DEFAULT_OBJECT_THRESHOLD = 0.25
DEFAULT_TEMPORAL_TOLERANCE = 0.25
DEFAULT_AUTO_ACCEPT_THRESHOLD = 0.85


def load_video_frames_at_fps(
    video_path: str | Path,
    fps: float = DEFAULT_FRAME_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a video once and return RGB frames plus timestamps."""
    if fps <= 0:
        raise ValueError("fps must be positive")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    source_fps = float(reader.get_avg_fps()) or 30.0
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video_path}")

    duration = total_frames / source_fps
    times = np.arange(0.0, duration, 1.0 / fps)
    if len(times) == 0:
        times = np.array([0.0])

    indices = np.clip(
        np.round(times * source_fps).astype(np.int64),
        0,
        total_frames - 1,
    )
    indices, unique_idx = np.unique(indices, return_index=True)
    times = times[unique_idx]
    frames = reader.get_batch(indices).asnumpy()
    return frames, times


def build_action_windows(
    frames: np.ndarray,
    times: Sequence[float],
    window_seconds: float = DEFAULT_ACTION_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_ACTION_STRIDE_SECONDS,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> list[dict]:
    """Create fixed temporal windows with uniformly sampled RGB frames."""
    if window_seconds <= 0 or stride_seconds <= 0 or num_frames <= 0:
        raise ValueError("window_seconds, stride_seconds and num_frames must be positive")

    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return []

    windows: list[dict] = []
    cursor = float(times[0])
    final_time = float(times[-1])

    while cursor <= final_time:
        end = cursor + window_seconds
        indices = np.where((times >= cursor) & (times <= end))[0]

        if len(indices) >= 2:
            chosen = np.linspace(indices[0], indices[-1], num_frames).round().astype(int)
            chosen = np.clip(chosen, 0, len(frames) - 1)
            windows.append(
                {
                    "start": float(times[chosen[0]]),
                    "end": float(times[chosen[-1]]),
                    "frames": frames[chosen],
                }
            )

        cursor += stride_seconds

    return windows


def temporal_overlap(
    action_start: float,
    action_end: float,
    object_time: float,
    tolerance: float = DEFAULT_TEMPORAL_TOLERANCE,
) -> bool:
    """Check whether an object timestamp is temporally associated with an action."""
    return action_start - tolerance <= object_time <= action_end + tolerance


def fuse_action_object(
    action_results: list[dict],
    object_results: list[dict],
    action_threshold: float = DEFAULT_ACTION_THRESHOLD,
    object_threshold: float = DEFAULT_OBJECT_THRESHOLD,
    temporal_tolerance: float = DEFAULT_TEMPORAL_TOLERANCE,
    auto_accept_threshold: float = DEFAULT_AUTO_ACCEPT_THRESHOLD,
) -> list[dict]:
    """Fuse action/object predictions and assign review status.

    Action-only predictions are retained as human-review candidates instead of
    being silently discarded when no object is found.
    """
    fused: list[dict] = []

    for action in action_results:
        action_conf = float(action["action_confidence"])
        if action_conf < action_threshold:
            continue

        matching = [
            obj
            for obj in object_results
            if float(obj["confidence"]) >= object_threshold
            and temporal_overlap(
                float(action["start"]),
                float(action["end"]),
                float(obj["timestamp"]),
                tolerance=temporal_tolerance,
            )
        ]

        if not matching:
            fused.append(
                {
                    "start": float(action["start"]),
                    "end": float(action["end"]),
                    "action": action["action"],
                    "object": None,
                    "action_confidence": action_conf,
                    "object_confidence": 0.0,
                    "combined_confidence": action_conf,
                    "source": "xclip_transfer",
                    "review_status": "human_review",
                }
            )
            continue

        best_by_object: dict[str, dict] = {}
        for obj in matching:
            key = str(obj["object"])
            if key not in best_by_object or float(obj["confidence"]) > float(best_by_object[key]["confidence"]):
                best_by_object[key] = obj

        for object_name, obj in best_by_object.items():
            object_conf = float(obj["confidence"])
            combined = float(np.sqrt(action_conf * object_conf))
            fused.append(
                {
                    "start": float(action["start"]),
                    "end": float(action["end"]),
                    "action": action["action"],
                    "object": object_name,
                    "action_confidence": action_conf,
                    "object_confidence": object_conf,
                    "combined_confidence": combined,
                    "source": "xclip_transfer+yoloe",
                    "review_status": (
                        "auto_accepted"
                        if combined >= auto_accept_threshold
                        else "human_review"
                    ),
                }
            )

    return fused


def run_parallel_video(
    video_path: str | Path,
    action_model: Any,
    object_model: Any,
    frame_fps: float = DEFAULT_FRAME_FPS,
    action_window_seconds: float = DEFAULT_ACTION_WINDOW_SECONDS,
    action_stride_seconds: float = DEFAULT_ACTION_STRIDE_SECONDS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    batch_size: int = 4,
    object_conf: float = DEFAULT_OBJECT_THRESHOLD,
    action_threshold: float = DEFAULT_ACTION_THRESHOLD,
    temporal_tolerance: float = DEFAULT_TEMPORAL_TOLERANCE,
    auto_accept_threshold: float = DEFAULT_AUTO_ACCEPT_THRESHOLD,
) -> dict:
    """Run both model branches over one video and return fused annotations."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    frames, times = load_video_frames_at_fps(video_path, frame_fps)
    windows = build_action_windows(
        frames,
        times,
        window_seconds=action_window_seconds,
        stride_seconds=action_stride_seconds,
        num_frames=num_frames,
    )

    def action_worker() -> list[dict]:
        results: list[dict] = []
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start : start + batch_size]
            frame_batch = [window["frames"] for window in batch_windows]

            if hasattr(action_model, "predict_batch"):
                predictions = action_model.predict_batch(frame_batch)
            else:
                predictions = [action_model.predict(frames) for frames in frame_batch]

            for window, pred in zip(batch_windows, predictions):
                results.append(
                    {
                        "start": window["start"],
                        "end": window["end"],
                        "action": pred["action"],
                        "action_confidence": pred["confidence"],
                        "action_probabilities": pred.get("probabilities", {}),
                        "source": pred.get("source", "xclip_transfer"),
                    }
                )
        return results

    def object_worker() -> list[dict]:
        results: list[dict] = []
        for frame, timestamp in zip(frames, times):
            for detection in object_model.predict(frame, conf=object_conf):
                detection = dict(detection)
                detection["timestamp"] = float(timestamp)
                results.append(detection)
        return results

    with ThreadPoolExecutor(max_workers=2) as executor:
        action_future = executor.submit(action_worker)
        object_future = executor.submit(object_worker)
        action_results = action_future.result()
        object_results = object_future.result()

    annotations = fuse_action_object(
        action_results,
        object_results,
        action_threshold=action_threshold,
        object_threshold=object_conf,
        temporal_tolerance=temporal_tolerance,
        auto_accept_threshold=auto_accept_threshold,
    )

    return {
        "video": Path(video_path).name,
        "action_predictions": action_results,
        "object_detections": object_results,
        "annotations": annotations,
    }
