import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from records import AtlasStore
from records.export import export_dataset
from records.test_store import photo
from vision.train import load_manifest


class ExportTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = AtlasStore(self.root / "atlas.sqlite3")
        self.photo = self.store.add_photo(photo(), "Identifying street")
        self.box = [0.1, 0.1, 0.8, 0.9]
        self.rid = self.store.add_region(self.photo, self.box)
        self.key = b"test-only-reviewer-key"

    def export(self, spec):
        return export_dataset(self.store, spec, self.root / "out", self.key)

    def test_conflict_roundtrip(self):
        store = self.store
        self.assertEqual(store.add_region(self.photo, self.box), self.rid)
        for name, verdict in [
            ("One", "present"),
            ("Two", "absent"),
            ("Three", "uncertain"),
        ]:
            store.report_region(self.rid, name, verdict)
        self.assertTrue(store.region_summaries(self.photo)[0]["conflict"])
        self.assertEqual(store.list_photos()[0]["report_count"], 3)
        manifest = self.export({str(self.photo): {"street": "G01", "split": "train"}})
        rows, _ = load_manifest(manifest)
        self.assertEqual(set(rows[0]["votes"].values()), {1, 0, None})
        self.assertEqual(rows[0]["box"], self.box)
        self.assertNotIn("one", manifest.read_text())
        store.report_region(self.rid, "Two", "present")
        self.assertFalse(store.region_summaries(self.photo)[0]["conflict"])

    def test_validation_and_failed_export_is_atomic(self):
        for invalid in [(0, 0, 0, 1), (0.1, 0.9, 0.8, 0.1), (0, 0, 0.00001, 0.00001)]:
            with self.assertRaises(ValueError):
                self.store.add_region(self.photo, invalid)
        with self.assertRaises(ValueError):
            self.export({str(self.photo): {"street": "N", "split": "test"}})
        self.assertFalse((self.root / "out").exists())
        other = self.store.add_photo(photo(), "Other viewpoint title")
        rid = self.store.add_region(other, self.box)
        for group in ["north", "another"]:
            with self.assertRaises(ValueError):
                self.export(
                    {
                        str(self.photo): {"street": "North", "split": "train"},
                        str(other): {
                            "street": group,
                            "split": "test",
                            "truth": {str(rid): 0},
                        },
                    }
                )
        valid = self.export(
            {
                str(self.photo): {
                    "street": "N",
                    "split": "test",
                    "truth": {str(self.rid): 0},
                }
            }
        )
        self.assertEqual(json.loads(valid.read_text())["truth"], 0)
