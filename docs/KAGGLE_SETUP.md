# Kaggle setup

The final notebook is primarily an executable Kaggle experiment because the EPIC-KITCHENS video files are accessed through Kaggle rather than redistributed in this repository.

## Public source links

- Assessment annotation dataset: https://www.kaggle.com/datasets/amansandhu408/assessment/data
- EPIC-KITCHENS P1 video subset: https://www.kaggle.com/datasets/ldthanh/epic-kitchens-100-p1
- Assessment Kaggle notebook: https://www.kaggle.com/code/amansandhu408/assessment-labeller

## Notebook inputs

For the executed pilot, the notebook attached the EPIC-KITCHENS P1 video partition plus the assessment annotation/demo dataset. The code is written so additional EPIC-KITCHENS parts can be attached later.

The notebook performs a leakage audit before source-model construction and excludes participants `P01`, `P04`, and `P18` from the supervised source pool. The ten assessment videos therefore remain outside supervised model training and validation.

Do not copy the full EPIC-KITCHENS corpus into the working directory. Use the linked public source and attach only the partitions required for the experiment.
