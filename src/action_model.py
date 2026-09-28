"""X-CLIP action-recognition utilities for the Labellerr assessment.

The module loads a frozen pretrained X-CLIP backbone and a small trained
7-class classification head produced by the assessment notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


ACTIONS = [
    "OPENS",
    "PICKS UP",
    "PUT DOWN",
    "POUR IN",
    "WALKS",
    "WASHING",
    "COOKING",
]


class ActionHead(nn.Module):
    """Small transfer-learning head trained on cached X-CLIP features."""

    def __init__(self, input_dim: int, num_classes: int = len(ACTIONS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class XClipActionRecognizer:
    """Frozen X-CLIP representation + trained action head."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        backbone_name: str = "microsoft/xclip-base-patch16",
        device: str | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Action-head checkpoint not found: {checkpoint_path}. "
                "Run the assessment notebook first to generate weights/action_head.pt."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        expected_actions = checkpoint.get("actions", ACTIONS)
        if list(expected_actions) != ACTIONS:
            raise ValueError(
                "The checkpoint action ontology does not match the project ontology."
            )

        backbone_name = checkpoint.get("backbone", backbone_name)
        self.processor = AutoProcessor.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name).to(self.device).eval()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        feature_dim = int(checkpoint["feature_dim"])
        self.head = ActionHead(feature_dim, len(ACTIONS)).to(self.device)
        self.head.load_state_dict(checkpoint["state_dict"])
        self.head.eval()

    @torch.inference_mode()
    def predict_batch(self, frame_windows: Sequence[Sequence[np.ndarray]]) -> list[dict]:
        """Predict a batch of temporal RGB-frame windows."""
        if len(frame_windows) == 0:
            return []

        inputs = self.processor(
            videos=[list(frames) for frames in frame_windows],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.backbone.get_video_features(pixel_values=pixel_values)
        features = features.pooler_output if hasattr(features, "pooler_output") else features[1]

        probs = torch.softmax(self.head(features.float()), dim=-1)
        probs = probs.detach().cpu().numpy()

        results: list[dict] = []
        for row in probs:
            index = int(np.argmax(row))
            results.append({
                "action": ACTIONS[index],
                "confidence": float(row[index]),
                "probabilities": {
                    ACTIONS[i]: float(row[i]) for i in range(len(ACTIONS))
                },
                "source": "xclip_transfer",
            })
        return results

    @torch.inference_mode()
    def predict(self, frames: Sequence[np.ndarray]) -> dict:
        """Predict the action for one temporal RGB-frame window."""
        if len(frames) == 0:
            raise ValueError("Action prediction requires at least one frame.")
        return self.predict_batch([frames])[0]
