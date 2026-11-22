import numpy as np
import pytest
from PIL import Image

from vision import field


def test_class_threshold_clipping_and_invalid_boxes():
    raw = np.array(
        [
            [0, 7, 0.95, -0.5, 0.2, 1.2, 0.8],  # car
            [0, 7, 0.49, 0.1, 0.2, 0.6, 0.8],  # weak
            [0, 15, 0.99, 0.1, 0.2, 0.6, 0.8],  # person, excluded
            [0, 7, 0.98, 0.7, 0.2, 0.6, 0.8],  # inverted bounds
            [0, 7, np.nan, 0.1, 0.2, 0.6, 0.8],
        ]
    )
    assert field._proposals(raw) == [
        {"label": "car", "score": 0.95, "box": [0, 0.2, 1, 0.8]}
    ]


def test_multiview_remaps_and_suppresses(monkeypatch):
    sizes = []

    def mock(image):
        sizes.append(image.size)
        if len(sizes) in (1, 2):
            # First two detections cover the same top-left pixel region.
            x = 32 / image.width
            y = 32 / image.height
            return [{"label": "car", "score": 0.9, "box": [0, 0, x, y]}]
        return []

    monkeypatch.setattr(field, "_infer_crop", mock)
    out = field.detect_objects(Image.new("RGB", (192, 128)))
    assert len(sizes) == 6
    assert len(out) == 1
    assert out[0]["box"] == pytest.approx([0, 0, 32 / 192, 0.25])


def test_weight_file_guard(tmp_path):
    (tmp_path / "deploy.prototxt").write_text("unexpected")
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="SHA-256"):
        field._network(tmp_path)


def test_prepared_field_sample():
    from pathlib import Path

    if not field.available():
        pytest.skip("Prepare optional MobileNet-SSD weights to run CPU inference")
    with Image.open(
        Path(__file__).resolve().parents[1] / "data_tools/field-example.jpg"
    ) as photo:
        hits = field.detect_objects(photo)
    assert any(hit["label"] == "motorcycle" for hit in hits)
