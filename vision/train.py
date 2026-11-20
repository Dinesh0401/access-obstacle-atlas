"""Fit a CNN on independently reported, possibly conflicting tile labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .network import Network, rgb_array, sigmoid


def load_manifest(path):
    rows, images = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] not in {"train", "anchor", "test"} or not row["street"]:
            raise ValueError("Expected street and a valid split")
        votes = row["votes"]
        if not isinstance(votes, dict) or any(
            v not in [0, 1, None] for v in votes.values()
        ):
            raise ValueError("One yes/no/unknown observation per reviewer per tile")
        if row["split"] != "train" and row.get("truth") not in (0, 1):
            raise ValueError("Anchor and test rows need adjudicated truth")
        with Image.open(path.parent / row["image"]) as original:
            img = ImageOps.exif_transpose(original)
            x, y, right, bottom = row.get("box", [0, 0, 1, 1])
            if not (0 <= x < right <= 1 and 0 <= y < bottom <= 1):
                raise ValueError("Invalid normalized box")
            crop = img.crop(
                (x * img.width, y * img.height, right * img.width, bottom * img.height)
            )
            if min(crop.size) < 1:
                raise ValueError("Empty tile")
            images.append(rgb_array(crop))
        rows.append(row)
    for key in ["image", "street"]:
        seen = {}
        for row in rows:
            if seen.setdefault(row[key], row["split"]) != row["split"]:
                raise ValueError(f"Leakage across splits for {key}")
    return rows, np.stack(images)


def rater_rates(rows):
    counts = {}
    for row in rows:
        if row["split"] == "anchor":
            for reviewer, vote in row["votes"].items():
                if vote is not None:
                    table = counts.setdefault(
                        reviewer, np.array([[2.0, 1.0], [1.0, 2.0]])
                    )
                    table[row["truth"], vote] += 1
    return {key: value / value.sum(1, keepdims=True) for key, value in counts.items()}


def evidence(row, rates):
    # Weakly informative fallback for a reviewer without an anchor.
    fallback = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    result = 0.0
    for reviewer, vote in row["votes"].items():
        if vote is not None:
            table = rates.get(reviewer, fallback)
            result += np.log(table[1, vote] / table[0, vote])
    return result


def fit(rows, images, mode, seed, epochs):
    rates = rater_rates(rows)
    indices = [
        i
        for i, r in enumerate(rows)
        if r["split"] == "train" and any(v is not None for v in r["votes"].values())
    ]
    if not indices:
        raise ValueError("No observed training labels")
    x = images[indices]
    llr = np.array([evidence(rows[i], rates) for i in indices])
    fractions = np.array(
        [
            np.mean([v for v in rows[i]["votes"].values() if v is not None])
            for i in indices
        ]
    )
    majority = np.where(fractions > 0.5, 1.0, np.where(fractions < 0.5, 0.0, 0.5))
    net = Network.initialize(seed)
    rng = np.random.default_rng(seed)
    moment = [np.zeros_like(w) for w in net.weights]
    variance = [np.zeros_like(w) for w in net.weights]
    step = 0
    for _ in range(epochs):
        for batch in np.array_split(
            rng.permutation(len(x)), max(1, int(np.ceil(len(x) / 32)))
        ):
            if mode == "fusion":
                logits = net.forward(x[batch])[0]
                targets = sigmoid(logits + llr[batch])
            else:
                targets = majority[batch]
            grads = net.gradients(x[batch], targets.astype("f"))
            step += 1
            for w, m, v, g in zip(net.weights, moment, variance, grads):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                w -= (
                    0.003
                    * (m / (1 - 0.9**step))
                    / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
    return net


def metrics(truth, scores):
    y = np.asarray(truth, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Test requires positive and negative examples")
    # Group ties before calculating average precision.
    order = np.argsort(-scores, kind="stable")
    labels, sorted_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tp = np.cumsum(labels)[ends]
    ap = np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / y.sum()
    positive = scores >= 0.5
    return {
        "average_precision": float(ap),
        "recall_at_0.5": float(positive[y == 1].mean()),
        "false_positive_rate_at_0.5": float(positive[y == 0].mean()),
        "brier_score": float(np.mean((scores - y) ** 2)),
    }


def grouped_ap_difference(truth, scores, street_ids, seed, repeats=1000):
    """Paired percentile interval conditional on the two fitted checkpoints."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(street_ids))
    groups = [np.flatnonzero(np.asarray(street_ids) == street) for street in unique]
    if len(groups) < 2:
        return {"groups": len(groups), "interval": None, "valid_replicates": 0}
    y = np.asarray(truth)
    differences = []
    for _ in range(repeats):
        sample = np.concatenate(
            [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        )
        if len(np.unique(y[sample])) < 2:
            continue
        ap = [
            metrics(y[sample], scores[mode][sample])["average_precision"]
            for mode in ("majority", "fusion")
        ]
        differences.append(ap[1] - ap[0])
    interval = np.percentile(differences, [2.5, 97.5]).tolist() if differences else None
    return {
        "groups": len(groups),
        "interval": interval,
        "valid_replicates": len(differences),
    }


def evaluate(rows, images, seed, epochs, checkpoint):
    idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
    truth = [rows[i]["truth"] for i in idx]
    results, predictions = {}, {}
    for mode in ("majority", "fusion"):
        model = fit(rows, images, mode, seed, epochs)
        scores = sigmoid(model.forward(images[idx])[0])
        predictions[mode] = scores
        results[mode] = metrics(truth, scores)
        if mode == "fusion":
            model.save(checkpoint)
    return {
        "seed": seed,
        "epochs": epochs,
        "test_tiles": len(idx),
        "test_positives": sum(truth),
        "models": results,
        "paired_ap_difference_cluster_bootstrap_95pct": grouped_ap_difference(
            truth, predictions, [rows[i]["street"] for i in idx], seed
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                *load_manifest(args.manifest), args.seed, args.epochs, args.checkpoint
            ),
            indent=2,
        )
    )
