"""Optional CPU object proposals with a pinned PASCAL VOC MobileNet-SSD."""

import argparse
import hashlib
import json
import os
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps

REVISION = "bb17b6c3eef36d80be441ae8e5339be66e8e3b7a"
UPSTREAM = f"https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/{REVISION}"
FILES = {
    "deploy.prototxt": "2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608",
    "mobilenet_iter_73000.caffemodel": "52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7",
}
LABELS = {
    2: "bicycle",
    6: "bus",
    7: "car",
    9: "chair",
    11: "table",
    14: "motorcycle",
    16: "potted plant",
    18: "sofa",
}
FIELD_CARD = {
    "name": "MobileNet-SSD",
    "training_data": "PASCAL VOC 2007 and 2012",
    "status": "Pretrained object proposals",
    "threshold": 0.5,
    "scope": "Physical objects only; a reviewer must inspect path overlap.",
}
# OpenCV mutates the input blob and internal buffers of each shared network.
_INFERENCE_LOCK = Lock()


def model_dir():
    custom = os.getenv("ATLAS_FIELD_MODEL_DIR")
    return (
        Path(custom).expanduser()
        if custom
        else Path.home() / ".cache/access-obstacle-atlas/voc-ssd"
    )


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare():
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        dest = directory / name
        if dest.is_file() and _digest(dest) == expected:
            continue
        with NamedTemporaryFile(dir=directory, delete=False) as f:
            temporary = Path(f.name)
        try:
            with (
                urlopen(f"{UPSTREAM}/{name}", timeout=90) as src,
                temporary.open("wb") as dst,
            ):
                copyfileobj(src, dst)
            if _digest(temporary) != expected:
                raise ValueError(f"Unexpected SHA-256 digest for {name}")
            os.replace(temporary, dest)
        finally:
            temporary.unlink(missing_ok=True)
    _network.cache_clear()
    return directory


@lru_cache(maxsize=1)
def _network(directory):
    import cv2

    for name, expected in FILES.items():
        if _digest(directory / name) != expected:
            raise ValueError(f"Unexpected SHA-256 digest for {name}")
    return cv2.dnn.readNetFromCaffe(
        str(directory / "deploy.prototxt"),
        str(directory / "mobilenet_iter_73000.caffemodel"),
    )


def available():
    if find_spec("cv2") is None:
        return False
    try:
        _network(model_dir())
    except (OSError, ImportError, ValueError):
        return False
    return True


def _proposals(detections):
    items = []
    for _, label_id, score, left, top, right, bottom in detections.reshape(-1, 7):
        if not np.all(np.isfinite([label_id, score, left, top, right, bottom])):
            continue
        label = LABELS.get(int(label_id))
        if not label or not FIELD_CARD["threshold"] <= score <= 1:
            continue
        box = np.clip([left, top, right, bottom], 0, 1).tolist()
        if box[2] > box[0] and box[3] > box[1]:
            items.append({"label": label, "score": float(score), "box": box})
    return sorted(items, key=lambda item: -item["score"])


def _infer_crop(image):
    import cv2

    # The Caffe checkpoint expects BGR, pixel mean 127.5, scale 0.007843.
    blob = cv2.dnn.blobFromImage(
        np.asarray(image), 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True
    )
    net = _network(model_dir())
    with _INFERENCE_LOCK:
        net.setInput(blob)
        output = net.forward().copy()
    return _proposals(output)


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return w * h / (aa + ab - w * h)


def detect_objects(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    candidates = _infer_crop(image)
    # Overlapping square views retain object aspect ratio on portrait streetscapes.
    # A whole-frame pass also retains context for large objects.
    width, height = image.size
    size = round(min(width, height) * 2 / 3)
    if size >= 64:
        for dx, dy in [(0, 0), (1, 0), (0.5, 0.5), (0, 1), (1, 1)]:
            x, y = round(dx * (width - size)), round(dy * (height - size))
            for item in _infer_crop(image.crop((x, y, x + size, y + size))):
                left, top, right, bottom = item["box"]
                item["box"] = [
                    (x + left * size) / width,
                    (y + top * size) / height,
                    (x + right * size) / width,
                    (y + bottom * size) / height,
                ]
                candidates.append(item)
    kept = []
    for item in sorted(candidates, key=lambda candidate: -candidate["score"]):
        if all(
            item["label"] != other["label"] or _iou(item["box"], other["box"]) <= 0.4
            for other in kept
        ):
            kept.append(item)
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare", action="store_true", help="Fetch and verify optional weights"
    )
    parser.add_argument("--image", type=Path, help="Photo to analyze locally")
    args = parser.parse_args()
    if args.prepare:
        print(prepare())
    if args.image:
        with Image.open(args.image) as source:
            print(json.dumps(detect_objects(source), indent=2))
    if not args.prepare and not args.image:
        parser.error("Provide --prepare or --image")
