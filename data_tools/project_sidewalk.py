"""Normalize v3 Project Sidewalk raw obstacle labels for historical map review."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def normalize(collection: dict, city: str) -> dict:
    if collection.get("type") != "FeatureCollection":
        raise ValueError("Expected GeoJSON FeatureCollection")
    features = []
    used = set()
    for source in collection["features"]:
        properties = source["properties"]
        if properties["label_type"] != "Obstacle":
            continue
        label_id = properties["label_id"]
        if type(label_id) is not int or label_id in used:
            raise ValueError(f"Invalid or duplicate label identifier: {label_id}")
        geometry = source["geometry"]
        if geometry["type"] != "Point":
            raise ValueError(f"Expected point geometry for label {label_id}")
        lon, lat = geometry["coordinates"]
        if not all(map(math.isfinite, [lon, lat])) or abs(lon) > 180 or abs(lat) > 90:
            raise ValueError(f"Invalid geographic coordinates for label {label_id}")
        votes = {"Human": Counter(), "AI": Counter()}
        for validation in properties["validations"]:
            kind = validation["validator_type"]
            decision = validation["validation"]
            if kind not in votes or decision not in {"Agree", "Disagree", "Unsure"}:
                raise ValueError(f"Unrecognized validation for label {label_id}")
            votes[kind][decision] += 1
        used.add(label_id)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": f"{city}:{label_id}",
                    "label": "Obstacle",
                    "region": properties.get("region_name"),
                    "street_id": properties.get("street_edge_id"),
                    "capture_date": properties.get("image_capture_date"),
                    "report_time": properties["time_created"],
                    "votes": votes,
                },
            }
        )
    features.sort(key=lambda f: f["properties"]["id"])
    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geojson", type=Path)
    parser.add_argument(
        "--city", required=True, help="City namespace for local label IDs"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    normalized = normalize(json.loads(args.geojson.read_text()), args.city)
    args.out.write_text(json.dumps(normalized, indent=2) + "\n")


if __name__ == "__main__":
    main()
