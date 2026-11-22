"""Install the exact model and CPU JavaScript runtime used by the hosted photo review."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def prepare():
    for name, source in json.loads((ROOT / "assets.json").read_text()).items():
        destination = ROOT / "detector-assets" / name
        expected = source["sha256"]
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            continue
        with urlopen(source["url"], timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


if __name__ == "__main__":
    prepare()
