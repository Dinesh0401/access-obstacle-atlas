import hashlib
import json

import numpy as np
from PIL import Image

from vision.api import CHECKPOINT, DEFAULT_CARD, configuration, detect
from vision.network import Network
from vision.train import evidence, load_manifest, metrics, rater_rates


def test_convolution_gradient():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(3, 16, 16, 3)).astype("f")
    y = np.array([1.0, 0.4, 0.0], "f")
    model = Network.initialize(9)
    grads = model.gradients(x, y)

    def objective():
        z = model.forward(x)[0].astype("d")
        return np.mean(np.logaddexp(0, z) - y * z)

    for weight, grad in zip(model.weights, grads):
        index = np.unravel_index(np.abs(grad).argmax(), grad.shape)
        original = weight[index]
        weight[index] = original + 0.001
        high = objective()
        weight[index] = original - 0.001
        low = objective()
        weight[index] = original
        np.testing.assert_allclose(
            (high - low) / 0.002, grad[index], atol=0.00015, rtol=0.04
        )


def test_tied_scores_and_missing_votes():
    assert metrics([0, 1, 0, 1], np.ones(4))["average_precision"] == 0.5
    anchor = {"split": "anchor", "votes": {"r": 1}, "truth": 1}
    rates = rater_rates([anchor])
    assert evidence({"votes": {"r": None}}, rates) == 0
    assert evidence(anchor, rates) > 0


def test_manifest_rejects_street_leak(tmp_path):
    import pytest

    rows = []
    for split in ["train", "test"]:
        name = split + ".png"
        Image.new("RGB", (16, 16)).save(tmp_path / name)
        rows.append(
            {"image": name, "truth": 1, "votes": {}, "split": split, "street": "same"}
        )
    manifest = tmp_path / "tiles.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="street"):
        load_manifest(manifest)


def test_checkpoint_card_integrity(tmp_path):
    import pytest

    card = dict(
        DEFAULT_CARD,
        checkpoint=str(CHECKPOINT),
        sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    )
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card))
    assert configuration(path)[0] == CHECKPOINT
    card["sha256"] = "bad"
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="Checkpoint"):
        configuration(path)


def test_detector_returns_valid_proposals():
    for item in detect(Image.new("RGB", (32, 29), "orange")):
        left, top, right, bottom = item["box"]
        assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1
        assert 0.5 <= item["score"] <= 1


def test_grouped_bootstrap_pairs_same_samples():
    from vision.train import grouped_ap_difference

    y = [0, 1, 0, 1]
    p = np.array([0.4, 0.8, 0.2, 0.5])
    out = grouped_ap_difference(
        y, {"majority": p, "fusion": p}, ["a", "a", "b", "b"], 4, 50
    )
    assert out == {"groups": 2, "interval": [0.0, 0.0], "valid_replicates": 50}
