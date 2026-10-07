# Rover Rescue — project rubric

Total: **100 points**. The same criteria apply to Colab, Jupyter, and Python projects. Grade behavior and evidence, not notebook structure or variable names.

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Learning system and method choices | 25 | Correct use of standardization, k-means, and 1-NN; meaningful calibration-based terrain mapping; choices explained. Library and custom implementations are equally eligible. |
| Representation and interpretation | 15 | Sensor-scale analysis (4), correctly fitted PCA and 95% component count (6), and a useful PCA figure with a supported interpretation (5). |
| Experimental design and evaluation | 25 | At least three configurations with meaningful scaling/PCA/model comparisons (8); a common held-out evaluation and correct fitting/label boundaries (9); correct dimensions, accuracy, macro F1, sand recall, and practice outcomes (8). |
| Deployment and reasoning | 10 | Evidence-based model choice (4), reuse of fitted transforms/predictor and a predicted mission terrain map (4), and a documented practice deployment (2). |
| Clear report | 20 | Focused question and rationale (4), readable table/figure and evidence-based conclusions (8), diagnosis/limitations including sand risk (5), and a clear evaluation/generalization explanation (3). Maximum two pages. |
| Reproducibility | 5 | Dependencies and exact run instructions; fresh execution reproduces evidence and mission predictions. Notebook and script submissions are equally eligible. |

Award partial credit for correct components even when later integration fails. An imperfect classifier or stuck rover can earn full credit when the implementation, evaluation, and reasoning meet the criteria. Extra experiments earn credit through stronger evidence, not their number alone. Optional features do not substitute for the required methods.

Assess claims against the submitted results. Do not require exact PCA signs, cluster numbering, a particular random seed, or agreement with a reference model. Inspect splitting and fitting boundaries directly. A validation score used for selection must not be presented as an untouched test score.
