"""Portable extensions of the mounted native NumPy/Adam checkpoint."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

import numpy as np

try:
    from .reference.data import flatten
    from .reference.model import Model
except ModuleNotFoundError as exc:
    if exc.name != "contest_engine.reference":
        raise
    from vendor.reference.A.baselines.data import flatten
    from vendor.reference.A.baselines.model import Model


TASKS = {"A": "classification", "B": "regression", "C": "regression"}
WIDTHS = {"A": 128, "B": 34, "C": 24}
NATIVE = "checkpoint.npz"
EXTENSION = "extension.npz"
CLASSES = ("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene")
ACTIONS = {
    "A": ("decay_pair", "decay_triad", "adam_small", "adam_tiny"),
    "B": ("residual", "residual_light", "adam_small", "adam_tiny"),
    "C": ("residual_light", "residual", "adam_small", "adam_tiny"),
}


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def input_vector(inputs, native):
    paths, values = flatten(inputs)
    if paths != native.meta["paths"]:
        raise ValueError("input schema/order differs from the mounted checkpoint")
    return values


def source_rows(rows, native):
    x, y = [], []
    classes = native.meta["classes"]
    for row in rows:
        x.append(input_vector(row["inputs"], native))
        target = row["answer"]
        if classes:
            target = classes.index(target)
        elif type(target) not in (int, float) or not np.isfinite(target) or target < 0:
            raise ValueError("regression train label must be finite and nonnegative")
        y.append(target)
    if not x:
        raise ValueError("phase training projection is empty")
    return np.asarray(x, dtype=np.float64), np.asarray(y)


def decay_view(x, floor):
    # Existing v18/v19 representation: log of the three decay responses per sensor.
    decay = x.reshape(-1, 16, 8)[:, :, 5:8]
    logs = np.log(np.maximum(np.abs(decay), floor))
    return (logs - logs.mean(axis=2, keepdims=True)).reshape(-1, 48)


def tree_output(x, side):
    children, values = side["children"], side["values"]
    trees = np.arange(len(children))[None, :]
    result = []
    for offset in range(0, len(x), 2048):
        batch = x[offset:offset + 2048]
        positions = np.zeros((len(batch), len(children)), dtype=np.intp)
        for _ in range(int(side["depth"]) + 1):
            feature = side["features"][trees, positions]
            threshold = side["thresholds"][trees, positions]
            branch = (batch[np.arange(len(batch))[:, None], feature] > threshold).astype(np.intp)
            positions = children[trees, positions, branch]
        result.append(float(side["intercept"]) + values[trees, positions].sum(axis=1))
    return np.concatenate(result)


@dataclass
class Predictor:
    dataset: str
    native: Model
    extension: dict | None = None

    def predict(self, x):
        if self.extension is None:
            return self.native.predict(x)
        side = self.extension
        if self.dataset == "A":
            scores = self.native.predict(x)
            labels = scores.argmax(axis=1)
            view = (decay_view(x, side["floor"]) - side["mean"]) / side["scale"]
            support = side["support"]
            distance = (view * view).sum(1)[:, None] + (support * support).sum(1) - 2 * view @ support.T
            margin = np.exp(-float(side["gamma"]) * np.maximum(distance, 0.)) @ side["dual"] + float(side["bias"])
            eligible = np.isin(labels, side["gate"])
            chosen = np.where(eligible & (margin > 0), 3, labels)
            negatives = side["gate"][side["gate"] != 3]
            rejected = (labels == 3) & (margin <= 0)
            chosen[rejected] = negatives[scores[rejected][:, negatives].argmax(axis=1)]
            changed = np.flatnonzero(chosen != labels)
            scores[changed, chosen[changed]], scores[changed, labels[changed]] = (
                scores[changed, labels[changed]], scores[changed, chosen[changed]].copy())
            return scores
        hidden, logits = self.native.forward(self.native.normalized(x))
        residual = tree_output(np.column_stack((x, hidden)), side)
        corrected = logits[:, 0] + float(side["strength"]) * residual
        return np.maximum(0., corrected * self.native.meta["target_scale"] + self.native.meta["target_mean"])


def load(native_path, extension_path=None, dataset=None):
    native = Model.load(native_path)
    inferred = next((key for key, size in WIDTHS.items() if size == len(native.meta["paths"])), None)
    if inferred is None or (dataset is not None and dataset != inferred) or native.meta["task"] != TASKS[inferred]:
        raise ValueError("unexpected industrial checkpoint schema")
    if inferred == "A" and tuple(native.meta["classes"]) != CLASSES:
        raise ValueError("unexpected gas order")
    side = None
    if extension_path is not None:
        with np.load(extension_path, allow_pickle=False) as arrays:
            side = {key: arrays[key] for key in arrays.files}
        if str(side.pop("dataset")) != inferred or str(side.pop("native_sha256")) != sha256_file(native_path):
            raise ValueError("extension does not bind to this mounted checkpoint")
    return Predictor(inferred, native, side)


def save(predictor, folder, frozen_native=None):
    native_path = folder / NATIVE
    if frozen_native is None:
        predictor.native.save(native_path)
    else:
        shutil.copyfile(frozen_native, native_path)
    if predictor.extension is not None:
        np.savez_compressed(folder / EXTENSION, dataset=np.array(predictor.dataset),
                            native_sha256=np.array(sha256_file(native_path)), **predictor.extension)


def fit_decay(x, y, gate):
    from sklearn.svm import SVC

    keep = np.isin(y, [0, 3])
    if len(np.unique(y[keep])) != 2:
        raise ValueError("decay fit needs both ethanol and acetaldehyde in authorized train")
    x, y = x[keep], y[keep] == 3
    floor = np.maximum(np.median(np.abs(x), axis=0).reshape(16, 8)[:, 5:8], 1e-8) * 1e-6
    v = decay_view(x, floor)
    mean, scale = v.mean(axis=0), np.maximum(v.std(axis=0), 1e-6)
    gamma = 1.0 / v.shape[1]
    fitted = SVC(C=2.0, gamma=gamma, class_weight="balanced", cache_size=64).fit((v - mean) / scale, y)
    return {"floor": floor, "mean": mean, "scale": scale, "gamma": np.array(gamma),
            "bias": np.array(fitted.intercept_[0]), "support": fitted.support_vectors_,
            "dual": fitted.dual_coef_[0], "gate": np.asarray(gate)}


def fit_residual(native, x, y, light):
    from sklearn.ensemble import HistGradientBoostingRegressor

    hidden, logits = native.forward(native.normalized(x))
    target = (y - native.meta["target_mean"]) / native.meta["target_scale"] - logits[:, 0]
    view = np.column_stack((x, hidden))
    depth, rounds, strength = (3, 20, .65) if light else (5, 36, .9)
    limit = min(31, 2 ** depth)
    fitted = HistGradientBoostingRegressor(
        max_depth=depth, max_leaf_nodes=limit, max_iter=rounds, learning_rate=.1,
        min_samples_leaf=20, l2_regularization=4., early_stopping=False, random_state=native.meta["seed"],
    ).fit(view, target)
    size = 2 * limit - 1
    features = np.zeros((rounds, size), dtype=np.intp)
    thresholds, values = np.zeros((rounds, size)), np.zeros((rounds, size))
    children = np.zeros((rounds, size, 2), dtype=np.intp)
    for i, predictors in enumerate(fitted._predictors):
        nodes = predictors[0].nodes
        n = len(nodes)
        leaf = nodes["is_leaf"].astype(bool)
        features[i, :n] = nodes["feature_idx"]
        thresholds[i, :n] = nodes["num_threshold"]
        values[i, :n] = nodes["value"]
        children[i, :n, 0] = np.where(leaf, np.arange(n), nodes["left"])
        children[i, :n, 1] = np.where(leaf, np.arange(n), nodes["right"])
    side = {"features": features, "values": values, "thresholds": thresholds, "children": children,
            "depth": np.array(depth), "strength": np.array(strength), "intercept": np.array(fitted._baseline_prediction.item())}
    check = view[:128]
    np.testing.assert_allclose(tree_output(check, side), fitted.predict(check), atol=1e-12, rtol=1e-12)
    return side


def train_action(parent, action, x, y):
    native = parent.native
    before = native.meta["steps"]
    if action in ("adam_small", "adam_tiny"):
        factor = .25 if action == "adam_small" else .08
        original_rate = native.meta["learning_rate"]
        native.meta["learning_rate"] = original_rate * factor
        try:
            loss = native.fit(x, y, 1)[0]
        finally:
            native.meta["learning_rate"] = original_rate
        return parent, {"epochs": 1, "lr_factor": factor, "steps": native.meta["steps"] - before, "train_loss": loss}
    if action in ("decay_pair", "decay_triad") and parent.dataset == "A":
        gate = [0, 3, 5] if action == "decay_triad" else [0, 3]
        side = fit_decay(x, y, gate)
        return Predictor(parent.dataset, native, side), {"epochs": 0, "solver_fits": 1, "gate": gate, "support": len(side["support"])}
    if action in ("residual", "residual_light") and parent.dataset in ("B", "C"):
        side = fit_residual(native, x, y, light=action == "residual_light")
        return Predictor(parent.dataset, native, side), {"epochs": 0, "solver_fits": 1, "trees": len(side["children"]),
                                                        "strength": float(side["strength"]), "depth": int(side["depth"])}
    raise ValueError("action is not available for this task")
