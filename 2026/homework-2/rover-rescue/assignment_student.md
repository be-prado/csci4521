# Homework 2 — Rover Rescue

Build a learning system that predicts **firm**, **rough**, or **sand** terrain from eight sensor measurements. The supplied simulator handles movement and path planning. Investigate how your model choices affect predictions and the rover's outcome.

## Getting started

Clone the repository and follow the README to run `run_demo.py`. Everything needed to begin is included: `rover_starter.py`, a small runnable example, requirements, and `demo_data/`. Use that supplied dataset for your experiments; you do not collect or generate your own data. Create your own Colab/Jupyter notebook or Python file for your solution.

## Your investigation

Use standardization, PCA, k-means, and 1-nearest-neighbor classification. You may use existing libraries or your own implementations. Use three clusters for the terrain classifier and calibration-training labels to attach terrain names to clusters.

Compare at least three configurations that together investigate scaling, a PCA representation, and cluster-based versus 1-NN classification. Choose useful controlled comparisons; add configurations if needed to support your conclusions. Report dimensions, validation accuracy, macro F1, sand recall, and practice rover outcome. Include a PCA visualization and the component count needed to retain 95% of the variance. Explain the consequences of missing sand.

Choose a fitted model using your evidence. Show its practice deployment and predicted mission terrain map. Explain one failure, surprise, tie, or limitation. Imperfect predictions or a stuck rover can still support an excellent analysis.

## Evaluate fairly

Use the fixed `split_data` split. Fit preprocessing and clustering only on fit rows. Use calibration-training labels to map clusters and train 1-NN. Evaluate configurations on the same validation rows. Reuse the selected fitted model for the mission without refitting on validation or mission observations.

Use the eight sensor columns as features, not coordinates. Only the simulator reads `practice_truth.bin`. Mission data has no terrain labels: use your trained model to predict them. Practice rover outcomes demonstrate deployment; they are not independent validation accuracy. Validation used for model selection is not an untouched test set.

## Submit through Gradescope

1. **Code and results:** submit a Google Colab link, Jupyter notebook, or Python project. We encourage notebooks with the completed run and outputs saved. Include your comparison table, readable figures, practice deployment, and predicted mission map. Make links accessible to the grading team. For scripts, include results as separate files. Give dependencies and brief run instructions so your results can be reproduced.
2. **Report:** a PDF of at most two pages, prepared however you prefer. State your question, support your conclusions with a compact table and at least one figure, explain your model choice and limitations, and describe your fitting/evaluation protocol.

Exporting mission predictions as `submission.csv` is optional. Do not submit the dataset or simulator truth files. Follow the course syllabus for collaboration, citations, and AI use. Check Gradescope for the due date and submission instructions.

## Grading

| Criterion | Points |
|---|---:|
| Learning system and method choices | 25 |
| Representation and interpretation | 15 |
| Experimental design and evaluation | 25 |
| Deployment and reasoning | 10 |
| Clear report | 20 |
| Reproducibility | 5 |

Credit reflects correct use of methods, fair comparisons, supported reasoning, and reproducibility. No points depend solely on hidden mission success.
