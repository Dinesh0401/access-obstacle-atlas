"""Small fully trainable convolutional patch scorer, with NumPy inference."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PATCH_SIZE = 16
GRID = 6


def rgb_array(image):
    return (
        np.asarray(
            image.convert("RGB").resize(
                (PATCH_SIZE, PATCH_SIZE), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 127.5
        - 1
    )


def tile_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    for row in range(GRID):
        for col in range(GRID):
            box = (col / GRID, row / GRID, (col + 1) / GRID, (row + 1) / GRID)
            bounds = tuple(
                round(c * d) for c, d in zip(box, (width, height, width, height))
            )
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                yield box, rgb_array(image.crop(bounds))


def windows(x):
    view = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    return view.transpose(0, 1, 2, 4, 5, 3).reshape(*view.shape[:3], -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class Network:
    weights: list

    @classmethod
    def initialize(cls, seed):
        rng = np.random.default_rng(seed)
        params = []
        for fan_in, fan_out in [(27, 8), (72, 12), (12, 1)]:
            params.extend(
                [
                    (
                        rng.standard_normal((fan_in, fan_out)) * np.sqrt(2 / fan_in)
                    ).astype("f"),
                    np.zeros(fan_out, "f"),
                ]
            )
        return cls(params)

    def forward(self, x):
        w1, b1, w2, b2, w3, b3 = self.weights
        first = windows(x)
        h1 = np.maximum(first @ w1 + b1, 0)
        # 14x14 -> 7x7 average pooling.
        pooled = h1.reshape(len(x), 7, 2, 7, 2, 8).mean(axis=(2, 4))
        second = windows(pooled)
        h2 = np.maximum(second @ w2 + b2, 0)
        feat = h2.mean(axis=(1, 2))
        logits = (feat @ w3 + b3).ravel()
        return logits, (first, h1, second, h2, feat)

    def gradients(self, x, targets):
        logits, (first, h1, second, h2, feat) = self.forward(x)
        dz = ((sigmoid(logits) - targets) / len(x)).reshape(-1, 1)
        w2, w3 = self.weights[2], self.weights[4]
        dh2 = (dz @ w3.T)[:, None, None] * (h2 > 0) / 25
        patches = (dh2 @ w2.T).reshape(len(x), 5, 5, 3, 3, 8)
        dp = np.zeros((len(x), 7, 7, 8), dtype="f")
        for i in range(3):
            for j in range(3):
                dp[:, i : i + 5, j : j + 5] += patches[:, :, :, i, j]
        dh1 = dp.repeat(2, 1).repeat(2, 2) * (h1 > 0) / 4
        return [
            first.reshape(-1, 27).T @ dh1.reshape(-1, 8),
            dh1.sum((0, 1, 2)),
            second.reshape(-1, 72).T @ dh2.reshape(-1, 12),
            dh2.sum((0, 1, 2)),
            feat.T @ dz,
            dz.sum(0),
        ]

    def save(self, path):
        np.savez_compressed(path, **{f"p{i}": p for i, p in enumerate(self.weights)})

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls([data[f"p{i}"] for i in range(6)])
