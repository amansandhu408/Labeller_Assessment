# Scalable Egocentric Video Labeling — Automation Note

## 1. Problem

Manual annotation requires an annotator to watch egocentric footage, locate action boundaries, identify the interacting object, and assign labels. As video volume grows, the work scales approximately linearly and a large fraction of annotator time can be spent on repetitive, visually obvious segments.

## 2. Proposed semi-automated workflow

The prototype uses a shared timestamped frame stream. Two independent model branches consume the same decoded frames:

```text
Video → frame sampling → [ X-CLIP action recognition || YOLOE open-vocabulary object detection ]
      → temporal alignment → confidence fusion → auto-accept OR human review → final annotations
```

The action branch predicts the seven project actions: `OPENS`, `PICKS UP`, `PUT DOWN`, `POUR IN`, `WALKS`, `WASHING`, and `COOKING`. The object branch uses YOLOE text prompts for `DOOR`, `FRIDGE`, `BOTTLE`, `GLASS`, `CUP`, and `DRAWER`. A detection is associated with an action when its timestamp falls inside, or close to, the action window. The prototype combines action and object confidence using the geometric mean and stores the result in a common JSON schema.

## 3. Human-in-the-loop

The goal is not to eliminate annotation review. High-confidence proposals can be accepted automatically, while low-confidence or ambiguous predictions are routed to an annotator. Human review is particularly useful for occlusion, unclear object identity, simultaneous actions, novel interactions, and temporal-boundary corrections. The reviewer can correct the action, object, timestamps, or reject a proposal. Those verified corrections can later become supervision for model calibration or retraining.

## 4. Quality controls

A production implementation should report action agreement, object agreement, temporal IoU, auto-accept rate, and human-review rate. Confidence thresholds should be calibrated on held-out data rather than selected only by intuition. In this assessment, the three human-verified clips are kept outside supervised training and validation; they are used only as reference/evaluation data.

## 5. Time and cost savings

The key efficiency gain is reducing the amount of video an annotator has to search manually. The measurable quantities are:

- Manual effort ≈ `N × manual_minutes_per_video`
- AI-assisted human effort ≈ `N × review_minutes_per_video`
- Effort reduction ≈ `1 − review_minutes_per_video / manual_minutes_per_video`

For an **illustrative, not measured** example, suppose manual labeling takes 6 minutes per video and reviewing AI proposals takes 2.5 minutes. Human effort would then fall by approximately 58%. At 1,000 videos that corresponds to roughly 3,500 minutes (58.3 hours) of human time avoided. Actual savings depend on video length, model quality, proposal density, threshold calibration, and the percentage of clips requiring review; therefore these values should be measured in a larger pilot before being used as production commitments.

## 6. Scaling path

For thousands or millions of clips, video decoding and model inference should be batched and distributed across workers. High-confidence annotations can move directly into the annotation store, while uncertain cases enter a human review queue. The review queue should surface the relevant timestamp, representative frames, predicted labels, confidence values, and an easy correction/rejection action.

The action and object branches are deliberately independent so that either model can later be replaced. Possible next steps include stronger temporal action models, object tracking/segmentation, pose estimation for interaction context, and a lightweight browser-based review interface.

## 7. Limitations

The action ontology is a project-level mapping of EPIC-KITCHENS verbs rather than a perfect semantic equivalence. After participant exclusion and physical-video filtering, `WALKS`, `COOKING`, and `POUR IN` are very sparse in the available pilot video subset. YOLOE is used with text prompts rather than fine-tuned on project-specific bounding boxes. The demonstration is therefore intended to validate the annotation workflow and automation design, not to claim production-grade accuracy.
