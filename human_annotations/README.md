# Human Annotations

This directory contains the three manually verified reference clips used to evaluate the automated annotation workflow.

| File | Purpose |
|---|---|
| `P01_03.json` | Human reference action + object intervals |
| `P01_04.json` | Human reference action + object intervals |
| `P04_19.json` | Human reference action + object intervals |

The assessment only requires two fully hand-labeled clips; three were labeled here to provide an additional held-out reference case.

## Annotation format

Each JSON file contains the video identifier, project ontology, and a list of annotations with:

- `start` / `end` timestamp in seconds
- canonical `action`
- interacting `object` when applicable
- `confidence: 1.0`
- `source: "human"`

These labels are evaluation/reference data only. They are not used to train or validate the supervised action model.
