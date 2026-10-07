"""Rover Rescue starter: data and simulator infrastructure only.

Write your learning methods and experiment code in your own script or notebook.
Import the public helpers from this file; leave simulator internals unchanged.
Run `python rover_starter.py PATH_TO_BUNDLE` to check data loading.
"""
from __future__ import annotations

import json
import heapq
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

_NAME_TO_INT = {"firm": 0, "rough": 1, "sand": 2}
_INT_TO_NAME = np.array(["firm", "rough", "sand"])


def load_bundle(bundle_dir: str | Path):
    bundle = Path(bundle_dir)
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    dev = pd.read_csv(bundle / "development.csv")
    cal = pd.read_csv(bundle / "calibration.csv")
    mission = pd.read_csv(bundle / "mission.csv")
    return dev, cal, mission, config


def _as_int_predictions(predictions: Iterable, n_expected: int) -> np.ndarray:
    arr = np.asarray(list(predictions) if not isinstance(predictions, np.ndarray) else predictions)
    arr = arr.reshape(-1)
    if len(arr) != n_expected:
        raise ValueError(f"Expected {n_expected} terrain predictions, got {len(arr)}.")

    if arr.dtype.kind in "OUS":
        out = []
        for value in arr:
            key = str(value).strip().lower()
            if key not in _NAME_TO_INT:
                raise ValueError(f"Unknown terrain label {value!r}; use firm, rough, or sand.")
            out.append(_NAME_TO_INT[key])
        return np.asarray(out, dtype=np.int8)

    arr = arr.astype(int)
    if not np.isin(arr, [0, 1, 2]).all():
        raise ValueError("Numeric terrain predictions must be 0=firm, 1=rough, or 2=sand.")
    return arr.astype(np.int8)


def _plan(pred: np.ndarray, shape: tuple[int, int], start: tuple[int, int], goal: tuple[int, int]):
    """Dijkstra planning on the STUDENT'S predicted terrain map."""
    h, w = shape
    queue = [(0.0, start)]
    dist = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}

    while queue:
        d, u = heapq.heappop(queue)
        if d != dist[u]:
            continue
        if u == goal:
            break
        y, x = u
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (y + dy, x + dx)
            if not (0 <= v[0] < h and 0 <= v[1] < w):
                continue
            terrain = int(pred[v[0] * w + v[1]])
            if terrain == 2:  # predicted sand is blocked
                continue
            step_cost = 1.0 if terrain == 0 else 2.4
            nd = d + step_cost
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(queue, (nd, v))

    if goal not in dist:
        return None, float("inf")

    path = []
    u = goal
    while u != start:
        path.append(u)
        u = prev[u]
    path.append(start)
    path.reverse()
    return path, dist[goal]


def _practice_truth(bundle_dir: str | Path) -> np.ndarray:
    with np.load(Path(bundle_dir) / "practice_truth.bin") as state:
        return state["terrain"].astype(np.int8)


def evaluate_practice(predictions, bundle_dir: str | Path) -> dict:
    _, _, _, cfg = load_bundle(bundle_dir)
    shape = tuple(cfg["grid_shape"])
    start = tuple(cfg["practice_start_yx"])
    goal = tuple(cfg["practice_goal_yx"])
    pred = _as_int_predictions(predictions, shape[0] * shape[1])
    truth = _practice_truth(bundle_dir)
    path, cost = _plan(pred, shape, start, goal)

    if path is None:
        return {"success": False, "status": "NO ROUTE", "steps": 0, "predicted_cost": None}

    for i, (y, x) in enumerate(path):
        if int(truth[y, x]) == 2:
            return {
                "success": False,
                "status": "ROVER STUCK IN SAND",
                "steps": i,
                "predicted_cost": round(float(cost), 2),
            }

    return {
        "success": True,
        "status": "MISSION SUCCESS",
        "steps": len(path) - 1,
        "predicted_cost": round(float(cost), 2),
    }


def show_practice_run(predictions, bundle_dir: str | Path, interval_ms: int = 90):
    """Return an HTML animation of the rover following a path chosen from predictions."""
    from IPython.display import HTML
    _, _, _, cfg = load_bundle(bundle_dir)
    shape = tuple(cfg["grid_shape"])
    h, w = shape
    start = tuple(cfg["practice_start_yx"])
    goal = tuple(cfg["practice_goal_yx"])
    pred = _as_int_predictions(predictions, h * w)
    truth = _practice_truth(bundle_dir)
    path, _ = _plan(pred, shape, start, goal)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    pred_map = pred.reshape(shape)
    cmap = ListedColormap(["#e9ecef", "#f4c95d", "#8d99ae"])
    ax.imshow(pred_map, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.scatter([start[1]], [start[0]], marker="s", s=90, label="start")
    ax.scatter([goal[1]], [goal[0]], marker="*", s=150, label="goal")
    rover, = ax.plot([], [], marker="o", markersize=10, linestyle="None")
    trail, = ax.plot([], [], linewidth=2)
    status = ax.set_title("Ready to deploy")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)

    if path is None:
        status.set_text("NO ROUTE: your predicted map blocks the goal")
        html = HTML(fig_to_html(fig))
        plt.close(fig)
        return html

    stop_index = len(path) - 1
    final_text = "MISSION SUCCESS"
    for i, (y, x) in enumerate(path):
        if int(truth[y, x]) == 2:
            stop_index = i
            final_text = "ROVER STUCK IN SAND"
            break
    shown_path = path[: stop_index + 1]

    def init():
        y, x = shown_path[0]
        rover.set_data([x], [y])
        trail.set_data([x], [y])
        status.set_text("Deploying rover...")
        return rover, trail, status

    def update(frame):
        travelled = shown_path[: frame + 1]
        ys = [p[0] for p in travelled]
        xs = [p[1] for p in travelled]
        rover.set_data([xs[-1]], [ys[-1]])
        trail.set_data(xs, ys)
        if frame == len(shown_path) - 1:
            status.set_text(final_text)
        else:
            status.set_text(f"Driving... step {frame}/{len(shown_path)-1}")
        return rover, trail, status

    anim = FuncAnimation(
        fig,
        update,
        frames=len(shown_path),
        init_func=init,
        interval=max(30, int(interval_ms)),
        blit=False,
        repeat=False,
    )
    html = HTML(anim.to_jshtml(default_mode="once"))
    plt.close(fig)
    return html


def fig_to_html(fig):
    """Static fallback when there is no path."""
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="Rover map" src="data:image/png;base64,{payload}" style="max-width:100%;height:auto">'


def save_submission(mission_predictions, bundle_dir: str | Path, filename: str | Path = "submission.csv") -> Path:
    """Validate and save final mission predictions in the format used by the grader."""
    bundle = Path(bundle_dir)
    mission = pd.read_csv(bundle / "mission.csv")
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    pred = _as_int_predictions(mission_predictions, len(mission))

    out = mission[["x", "y"]].copy()
    out["predicted_terrain"] = _INT_TO_NAME[pred]
    out.insert(0, "student_id", cfg["student_id"])
    path = Path(filename)
    out.to_csv(path, index=False)
    return path


# Shared evaluation split; no learning algorithms are supplied.
from sklearn.model_selection import train_test_split


def split_data(dev, calibration):
    """60 calibration-training labels, 30 validation labels in revision 2."""
    train, validation = train_test_split(calibration, test_size=1/3,
        stratify=calibration['terrain'], random_state=0)
    train, validation = train.reset_index(drop=True), validation.reset_index(drop=True)
    lookup = {(int(r.x), int(r.y)): i for i, r in dev.iterrows()}
    train_indices = np.array([lookup[(int(r.x), int(r.y))] for _, r in train.iterrows()])
    validation_indices = np.array([lookup[(int(r.x), int(r.y))] for _, r in validation.iterrows()])
    fit_indices = np.setdiff1d(np.arange(len(dev)), validation_indices)
    return train, validation, fit_indices, train_indices, validation_indices


def main():
    """Load data and print sizes; students design the learning pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_dir', type=Path, help='Directory containing your assigned CSVs and config.json')
    args = parser.parse_args()
    development, calibration, mission, config = load_bundle(args.bundle_dir)
    training, validation, fit_rows, train_rows, validation_rows = split_data(development, calibration)
    print(f"Development: {len(development)} rows; mission: {len(mission)} rows")
    print(f"Fit: {len(fit_rows)} rows; labeled training: {len(training)}; validation: {len(validation)}")
    print("Data loaded. Build your methods and experiments in your own script or notebook.")


if __name__ == '__main__':
    main()
