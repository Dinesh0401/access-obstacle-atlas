"""Region proposals from the bundled demo or an explicitly described checkpoint."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .network import Network, sigmoid, tile_image

BASE = Path(__file__).parent
DEFAULT_CARD = {
    "name": "Atlas Patch CNN",
    "training_data": "Procedural pavement and obstruction tiles",
    "status": "Synthetic demonstration only",
    "scope": "Grid-cell obstruction proposals; real-street accuracy unknown",
    "threshold": 0.5,
}


def configuration(config_path=None):
    if not config_path:
        return BASE / "demo.npz", DEFAULT_CARD.copy()
    source = Path(config_path)
    data = json.loads(source.read_text())
    path = source.parent / data["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != data["sha256"]:
        raise ValueError("Checkpoint differs from its model card")
    card = {key: data[key] for key in DEFAULT_CARD}
    if not all(
        isinstance(card[k], str) and card[k].strip()
        for k in ("name", "training_data", "status", "scope")
    ):
        raise ValueError("Model card requires training and evaluation provenance")
    if not 0 < card["threshold"] < 1:
        raise ValueError("Model threshold must be strictly between zero and one")
    return path, card


CHECKPOINT, MODEL_CARD = configuration(os.environ.get("ATLAS_MODEL_CONFIG"))


@lru_cache(maxsize=1)
def _network():
    return Network.load(CHECKPOINT)


def detect(image):
    tiles = list(tile_image(image))
    scores = sigmoid(_network().forward(np.stack([x for _, x in tiles]))[0])
    return sorted(
        (
            {"label": "obstruction", "score": float(score), "box": list(box)}
            for (box, _), score in zip(tiles, scores)
            if score >= MODEL_CARD["threshold"]
        ),
        key=lambda item: -item["score"],
    )
