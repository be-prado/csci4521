# Rover Rescue

Use eight sensor measurements to predict firm, rough, and sand terrain. The simulator handles the rover's movement. You build the learning system in your own notebook or Python file.

## Start locally

```bash
git clone https://github.com/be-prado/csci4521.git
cd csci4521/2026/homework-2/rover-rescue
python -m pip install -r requirements.txt
python run_demo.py
```

This loads the supplied data and runs a simple “all firm” prediction to check the simulator. Replace that baseline with your own model in your own file.

## Start in Colab

Create a blank notebook and run this cell:

```python
!git clone https://github.com/be-prado/csci4521.git
%cd csci4521/2026/homework-2/rover-rescue
!pip -q install -r requirements.txt
!python run_demo.py
```

Then write your solution in new cells. You can also use Jupyter or a Python script.

## Load the data

```python
from rover_starter import load_bundle, split_data

development, calibration, mission, config = load_bundle("demo_data")
calibration_train, validation, fit_indices, train_indices, validation_indices = split_data(development, calibration)
X = development[config["sensor_columns"]].to_numpy()
print(X.shape)
```

Use `demo_data/` for your experiments. It is a supplied synthetic dataset; you do not need to collect or generate data. Development and mission files contain sensor readings; calibration supplies labels for selected development locations. Coordinates identify locations and are not features. Keep the fixed split so your comparisons use the same validation observations.

## Run the rover with your predictions

Once your model produces one terrain label per development row:

```python
from rover_starter import evaluate_practice, show_practice_run

print(evaluate_practice(practice_predictions, "demo_data"))
# In a notebook:
display(show_practice_run(practice_predictions, "demo_data"))
# In a Python script, save the visualization instead:
# from pathlib import Path
# Path("practice_run.html").write_text(show_practice_run(practice_predictions, "demo_data").data)
```

Labels must be `firm`, `rough`, or `sand`, in CSV row order. Only the simulator reads `practice_truth.bin`. Apply your selected fitted model to `mission.csv` and show its predicted terrain map in your results.

See [the assignment](assignment_student.md) for the project and submission requirements. You choose your methods, code structure, and report format.
