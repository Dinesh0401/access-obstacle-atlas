"""Export explicitly selected regional observations for vision.train."""

import argparse
import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path

from .store import AtlasStore


def export_dataset(
    store: AtlasStore, assignments: dict, destination: Path, secret: bytes
) -> Path:
    """Create a fresh dataset atomically. Preserve ties and missing observations."""
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Supply a nonempty photo-ID to assignment object.")
    if len(secret) < 16:
        raise ValueError("Use at least 16 bytes for the reviewer pseudonymization key.")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    splits = {}
    image_splits = {}
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        output = Path(staging) / "dataset"
        (output / "images").mkdir(parents=True)
        with (
            store._connection() as db,
            (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
        ):
            db.execute(
                "BEGIN"
            )  # One snapshot even if field reports arrive during export.
            for photo_key, spec in assignments.items():
                if (
                    not isinstance(spec, dict)
                    or not isinstance(photo_key, str)
                    or not photo_key.isdecimal()
                ):
                    raise ValueError(
                        "Expected a numeric photo ID and an assignment object."
                    )
                photo_id = int(photo_key)
                if str(photo_id) != photo_key:
                    raise ValueError("Use canonical photo IDs without leading zeros.")
                split, street = spec.get("split"), spec.get("street")
                if (
                    split not in ("train", "anchor", "test")
                    or not isinstance(street, str)
                    or not street.strip()
                ):
                    raise ValueError(
                        "Assign a street group and train, anchor, or test split."
                    )
                street = street.strip().casefold()
                if splits.setdefault(street, split) != split:
                    raise ValueError(f"Street group {street} crosses splits.")
                photo = db.execute(
                    "SELECT image FROM photos WHERE id=?", (photo_id,)
                ).fetchone()
                if photo is None:
                    raise KeyError(photo_id)
                fingerprint = hashlib.sha256(photo["image"]).hexdigest()
                if image_splits.setdefault(fingerprint, split) != split:
                    raise ValueError("The same normalized image occurs across splits.")
                regions = db.execute(
                    "SELECT * FROM regions WHERE photo_id=? ORDER BY id", (photo_id,)
                ).fetchall()
                if not regions:
                    raise ValueError(f"Photo {photo_id} has no registered regions.")
                truth = spec.get("truth", {})
                expected = (
                    {str(r["id"]) for r in regions} if split != "train" else set()
                )
                if (
                    not isinstance(truth, dict)
                    or set(truth) != expected
                    or any(
                        type(v) is not int or v not in (0, 1) for v in truth.values()
                    )
                ):
                    raise ValueError(
                        f"Photo {photo_id} requires independent truth per anchor/test region only."
                    )
                image = f"images/{fingerprint}.jpg"
                (output / image).write_bytes(photo["image"])
                for region in regions:
                    rid = region["id"]
                    votes = {}
                    for reviewer, verdict in db.execute(
                        "SELECT reviewer,verdict FROM regional_reports WHERE region_id=? ORDER BY reviewer",
                        (rid,),
                    ):
                        alias = hmac.digest(secret, reviewer.encode(), "sha256").hex()
                        votes[alias] = {"present": 1, "absent": 0, "uncertain": None}[
                            verdict
                        ]
                    row = {
                        "image": image,
                        "box": [region[k] for k in ("x1", "y1", "x2", "y2")],
                        "region_id": rid,
                        "street": street,
                        "split": split,
                        "votes": votes,
                    }
                    if split != "train":
                        row["truth"] = truth[str(rid)]
                    manifest.write(json.dumps(row, allow_nan=False) + "\n")
        output.rename(destination)
    return destination / "manifest.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignments", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--db", type=Path, help="Default: ATLAS_DB or data/atlas.sqlite3"
    )
    args = parser.parse_args()
    key = os.environ.get("ATLAS_EXPORT_KEY", "").encode()
    path = export_dataset(
        AtlasStore(args.db),
        json.loads(args.assignments.read_text()),
        args.destination,
        key,
    )
    print(path)


if __name__ == "__main__":
    main()
