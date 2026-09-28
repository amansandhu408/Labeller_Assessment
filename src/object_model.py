"""YOLOE open-vocabulary object detection utilities.

YOLOE is configured with the six object prompts used by the assessment.
No project-specific bounding-box fine-tuning is assumed.
"""

from __future__ import annotations

from typing import Any

from ultralytics import YOLOE


OBJECT_PROMPTS = {
    "DOOR": "door",
    "FRIDGE": "refrigerator",
    "BOTTLE": "bottle",
    "GLASS": "drinking glass",
    "CUP": "cup",
    "DRAWER": "drawer",
}

PROMPT_TO_CANONICAL = {value: key for key, value in OBJECT_PROMPTS.items()}


class YOLOEObjectDetector:
    """YOLOE detector configured with assessment-specific text prompts."""

    def __init__(
        self,
        checkpoint: str = "yoloe-26s-seg.pt",
        device: str | None = None,
    ) -> None:
        self.model = YOLOE(checkpoint)
        self.model.set_classes(list(OBJECT_PROMPTS.values()))
        self.device = device

    def predict(self, frame_rgb: Any, conf: float = 0.25) -> list[dict]:
        """Detect prompted objects in one RGB frame."""
        kwargs = {"conf": conf, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device

        result = self.model.predict(frame_rgb, **kwargs)[0]
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names

        detections: list[dict] = []
        for box, score, cls_id in zip(boxes, confs, class_ids):
            prompt_name = str(names[int(cls_id)])
            canonical = PROMPT_TO_CANONICAL.get(prompt_name, prompt_name.upper())
            detections.append(
                {
                    "object": canonical,
                    "prompt_name": prompt_name,
                    "confidence": float(score),
                    "bbox_xyxy": [float(value) for value in box],
                }
            )

        return detections
