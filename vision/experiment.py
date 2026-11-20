"""Rebuild the procedural benchmark and bundled checkpoint."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .network import rgb_array
from .train import evaluate


def procedural_tile(rng, positive, *, geometry=False):
    box = None
    gray = int(rng.integers(75, 178))
    rgb = np.clip(
        gray + rng.normal(0, 7, (32, 32, 1)) + rng.uniform(-7, 7, 3), 0, 255
    ).astype("uint8")
    img = Image.fromarray(rgb)
    pen = ImageDraw.Draw(img)
    for _ in range(2):
        y = int(rng.integers(0, 32))
        pen.line((0, y, 32, y + int(rng.integers(-6, 7))), fill=(70, 70, 75))
    if positive:
        x, y = rng.integers(4, 16, 2).tolist()
        w, h = rng.integers(6, 16, 2).tolist()
        box = (x, y, x + w + 1, y + h + 1)
        color = [(94, 55, 40), (152, 90, 32), (45, 55, 89)][int(rng.integers(3))]
        pen.rectangle((x, y, x + w, y + h), fill=color)
        pen.line((x, y + h, x + w, y + h), fill=(38, 35, 32), width=2)
    elif rng.random() < 0.30:
        pen.line((2, 15, 28, 17), fill=(165, 130, 70), width=2)
    return (img, box) if geometry else rgb_array(img)


def dataset(seed=411):
    rng = np.random.default_rng(seed)
    rows, images = [], []
    reliability = [(0.90, 0.97), (0.76, 0.92), (0.82, 0.83), (0.48, 0.73)]
    for split, n in [("anchor", 180), ("train", 550), ("test", 250)]:
        for i in range(n):
            truth = int(rng.random() < (0.45 if split == "anchor" else 0.12))
            images.append(procedural_tile(rng, truth))
            votes = {}
            for j, (sensitivity, specificity) in enumerate(reliability):
                if rng.random() < (0.85 if split == "anchor" else 0.36):
                    votes[str(j)] = int(
                        rng.random() < (sensitivity if truth else 1 - specificity)
                    )
            rows.append(
                {
                    "street": f"{split}-{i // 10}",
                    "split": split,
                    "votes": votes,
                    "truth": truth,
                }
            )
    return rows, np.stack(images)


def localization_benchmark(model, scene_seed=702):
    """Score grid proposals against rendered object rectangles on new scenes."""
    from .network import GRID, sigmoid, tile_image

    rng = np.random.default_rng(scene_seed)
    gt, proposals = [], []
    for scene_index in range(20):
        scene = Image.new("RGB", (GRID * 32, GRID * 32))
        for row in range(GRID):
            for col in range(GRID):
                tile, box = procedural_tile(rng, rng.random() < 0.12, geometry=True)
                scene.paste(tile, (col * 32, row * 32))
                if box:
                    corners = [
                        u + offset for u, offset in zip(box, (col * 32, row * 32) * 2)
                    ]
                    gt.append((scene_index, np.array(corners) / (GRID * 32)))
        cells = list(tile_image(scene))
        scores = sigmoid(model.forward(np.stack([x for _, x in cells]))[0])
        proposals.extend(
            (float(score), scene_index, np.asarray(box))
            for score, (box, _) in zip(scores, cells)
        )
    results = {
        "scenes": 20,
        "ground_truth_objects": len(gt),
        "generator_seed": scene_seed,
    }
    for threshold in (0.1, 0.5):
        matched = set()
        true_positives = []
        for _, scene, box in sorted(proposals, key=lambda item: -item[0]):
            candidates = []
            for i, (target_scene, target) in enumerate(gt):
                if i in matched or target_scene != scene:
                    continue
                intersection = np.prod(
                    np.maximum(
                        0,
                        np.minimum(box[2:], target[2:])
                        - np.maximum(box[:2], target[:2]),
                    )
                )
                union = (
                    np.prod(box[2:] - box[:2])
                    + np.prod(target[2:] - target[:2])
                    - intersection
                )
                candidates.append((intersection / union, i))
            iou, idx = max(candidates, default=(0, -1))
            true_positives.append(iou >= threshold)
            if iou >= threshold:
                matched.add(idx)
        precision = np.cumsum(true_positives) / np.arange(1, len(proposals) + 1)
        results[f"object_ap_iou_{threshold}"] = float(
            np.sum(precision * np.array(true_positives)) / len(gt)
        )
    return results


if __name__ == "__main__":
    base = Path(__file__).parent
    rows, images = dataset()
    result = evaluate(rows, images, 9, 18, base / "demo.npz")
    from .network import Network

    result["synthetic_localization"] = localization_benchmark(
        Network.load(base / "demo.npz")
    )
    train = [r for r in rows if r["split"] == "train"]
    result.update(
        provenance="procedural synthetic-only patch classification",
        generator_seed=411,
        anchor_tiles=180,
        train_tiles=len(train),
        train_unobserved=sum(not r["votes"] for r in train),
        train_disputed=sum(len(set(r["votes"].values())) == 2 for r in train),
    )
    (base / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
