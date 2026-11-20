import json
import unittest
from pathlib import Path

from data_tools.project_sidewalk import normalize


class TestImport(unittest.TestCase):
    def test_preserves_disagreement_and_protects_identity(self):
        feature = {
            "geometry": {"type": "Point", "coordinates": [-77, 38]},
            "properties": {
                "label_id": 2,
                "label_type": "Obstacle",
                "time_created": "2020",
                "validations": [
                    {
                        "user_id": "private",
                        "validator_type": "Human",
                        "validation": "Agree",
                    },
                    {
                        "user_id": "private2",
                        "validator_type": "Human",
                        "validation": "Disagree",
                    },
                    {"user_id": "bot", "validator_type": "AI", "validation": "Unsure"},
                ],
            },
        }
        collection = {"type": "FeatureCollection", "features": [feature]}
        result = normalize(collection, "dc")
        props = result["features"][0]["properties"]
        self.assertEqual(props["votes"]["Human"], {"Agree": 1, "Disagree": 1})
        self.assertEqual(props["votes"]["AI"], {"Unsure": 1})
        self.assertNotIn("private", json.dumps(result))
        collection["features"].append(feature)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize(collection, "dc")

    def test_shipped_snapshot(self):
        data = json.loads(Path(__file__).with_name("sample-dc.json").read_text())
        self.assertEqual(len(data["features"]), 29)
        self.assertEqual(len({f["properties"]["id"] for f in data["features"]}), 29)
