import concurrent.futures
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from records import AtlasStore


def photo():
    buffer = BytesIO()
    exif = Image.Exif()
    exif[315] = "identifying information"
    Image.new("RGB", (24, 18), "white").save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


class AtlasStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = AtlasStore(Path(self.tmp.name) / "atlas.sqlite3")
        self.id = self.db.add_photo(photo(), "West crossing")

    def test_normalized_image_and_persistence(self):
        reopened = AtlasStore(self.db.path)
        self.assertEqual(reopened.list_photos()[0]["title"], "West crossing")
        with Image.open(BytesIO(reopened.get_image(self.id))) as image:
            self.assertFalse(image.getexif())
            self.assertEqual(image.size, (24, 18))

    def test_invalid_photo_coordinates_and_missing_photo(self):
        for args in [
            (b"garbage", "bad"),
            (photo(), "bad", 1, None),
            (photo(), "bad", float("nan"), 1),
        ]:
            with self.assertRaises(ValueError):
                self.db.add_photo(*args)
        with self.assertRaises(KeyError):
            self.db.report(-1, "A", "obstruction", "present")
        self.assertEqual(len(self.db.list_photos()), 1)

    def test_revisions_and_independent_attributes(self):
        self.db.report(self.id, "Alice", "obstruction", "present")
        self.db.report(self.id, " Bob ", "obstruction", "absent")
        self.db.report(self.id, "Alice", "ramp", "absent")
        self.assertEqual(self.db.summaries(self.id)[0]["status"], "conflicting")
        self.db.report(self.id, "BOB", "obstruction", "uncertain")
        row = self.db.list_photos()[0]
        self.assertEqual((row["report_count"], row["conflict_count"]), (3, 0))
        self.assertEqual(len(self.db.observations(self.id)), 3)

    def test_simultaneous_updates_preserve_one_current_vote(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda i: self.db.report(
                        self.id,
                        f"P{i % 3}",
                        "obstruction",
                        "present" if i % 2 else "absent",
                    ),
                    range(20),
                )
            )
        self.assertEqual(self.db.list_photos()[0]["report_count"], 3)
