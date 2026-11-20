"""SQLite persistence for normalized photos and amendable observer reports."""

from __future__ import annotations

import math
import os
import sqlite3
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    image BLOB NOT NULL,
    latitude REAL,
    longitude REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((latitude IS NULL) = (longitude IS NULL)),
    CHECK (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)
);
CREATE TABLE IF NOT EXISTS reports (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    obstacle TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('present','absent','uncertain')),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (photo_id, reviewer, obstacle)
);
CREATE TABLE IF NOT EXISTS regions (
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    UNIQUE(photo_id,x1,y1,x2,y2),
    CHECK(0<=x1 AND x1<x2 AND x2<=1 AND 0<=y1 AND y1<y2 AND y2<=1)
);
CREATE TABLE IF NOT EXISTS regional_reports (
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('present','absent','uncertain')),
    PRIMARY KEY(region_id,reviewer)
);
CREATE VIEW IF NOT EXISTS regional_evidence AS
    SELECT r.id,r.photo_id,r.x1,r.y1,r.x2,r.y2,
           count(CASE WHEN v.verdict='present' THEN 1 END) AS present,
           count(CASE WHEN v.verdict='absent' THEN 1 END) AS absent,
           count(CASE WHEN v.verdict='uncertain' THEN 1 END) AS uncertain
    FROM regions r LEFT JOIN regional_reports v ON r.id=v.region_id GROUP BY r.id;
CREATE VIEW IF NOT EXISTS evidence AS
    SELECT photo_id, obstacle, sum(verdict='present') AS present,
           sum(verdict='absent') AS absent, sum(verdict='uncertain') AS uncertain
    FROM reports GROUP BY photo_id, obstacle;
"""


def _text(value: str, field: str, limit: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise ValueError(f"{field} must contain 1–{limit} characters.")
    value = value.strip()
    if any(not c.isprintable() for c in value):
        raise ValueError(f"{field} contains unsupported control characters.")
    return value


def _coordinates(latitude, longitude):
    if latitude is None and longitude is None:
        return None, None
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Enter both latitude and longitude.") from None
    if not (
        math.isfinite(lat)
        and -90 <= lat <= 90
        and math.isfinite(lon)
        and -180 <= lon <= 180
    ):
        raise ValueError("Coordinates fall outside the map.")
    return lat, lon


def _normalize_image(payload: bytes) -> bytes:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= 10 * 1024 * 1024:
        raise ValueError("Choose a photo of up to 10 MiB.")
    try:
        with Image.open(BytesIO(payload), formats=["JPEG", "PNG", "WEBP"]) as image:
            if (
                image.width * image.height > 16_000_000
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError("Choose one still photo of up to 16 megapixels.")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
            rgb.info.clear()
            output = BytesIO()
            rgb.save(output, format="JPEG", quality=89)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise ValueError("Choose a readable JPEG, PNG, or WebP photo.") from None


class AtlasStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("ATLAS_DB", "data/atlas.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def add_photo(
        self, image_bytes: bytes, title: str, latitude=None, longitude=None
    ) -> int:
        title = _text(title, "Title", 120)
        latitude, longitude = _coordinates(latitude, longitude)
        image = _normalize_image(image_bytes)
        with self._connection() as db:
            return db.execute(
                "INSERT INTO photos(title,image,latitude,longitude) VALUES(?,?,?,?)",
                (title, image, latitude, longitude),
            ).lastrowid

    def list_photos(self) -> list[dict]:
        query = """
            SELECT p.id,p.title,p.created_at,p.latitude,p.longitude,
                   coalesce(sum(e.present+e.absent+e.uncertain),0) AS report_count,
                   coalesce(sum(e.present>0 AND e.absent>0),0) AS conflict_count
            FROM photos p LEFT JOIN (
                SELECT photo_id,present,absent,uncertain FROM evidence
                UNION ALL
                SELECT photo_id,present,absent,uncertain FROM regional_evidence
            ) e ON p.id=e.photo_id
            GROUP BY p.id ORDER BY conflict_count DESC,p.id DESC
        """
        with self._connection() as db:
            return [dict(row) for row in db.execute(query)]

    def get_image(self, photo_id: int) -> bytes:
        with self._connection() as db:
            row = db.execute(
                "SELECT image FROM photos WHERE id=?", (photo_id,)
            ).fetchone()
            if row is None:
                raise KeyError(photo_id)
            return row["image"]

    def report(self, photo_id: int, reviewer: str, obstacle: str, verdict: str) -> None:
        reviewer = _text(reviewer, "Reviewer", 80).casefold()
        obstacle = _text(obstacle, "Obstacle", 80).casefold()
        if verdict not in ("present", "absent", "uncertain"):
            raise ValueError("Choose present, absent, or uncertain.")
        with self._connection() as db:
            try:
                db.execute(
                    """INSERT INTO reports(photo_id,reviewer,obstacle,verdict)
                       VALUES(?,?,?,?) ON CONFLICT(photo_id,reviewer,obstacle)
                       DO UPDATE SET verdict=excluded.verdict,updated_at=CURRENT_TIMESTAMP""",
                    (photo_id, reviewer, obstacle, verdict),
                )
            except sqlite3.IntegrityError as exc:
                if exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY:
                    raise KeyError(photo_id) from exc
                raise

    def observations(self, photo_id: int) -> list[dict]:
        with self._connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT reviewer,obstacle,verdict FROM reports WHERE photo_id=? ORDER BY obstacle,reviewer",
                    (photo_id,),
                )
            ]

    def summaries(self, photo_id: int) -> list[dict]:
        query = """
            SELECT obstacle,present,absent,uncertain,
                   (present>0 AND absent>0) AS conflict,
                   CASE WHEN present>0 AND absent>0 THEN 'conflicting'
                        WHEN present>0 THEN 'reported present'
                        WHEN absent>0 THEN 'reported absent'
                        ELSE 'uncertain' END AS status
            FROM evidence WHERE photo_id=? ORDER BY conflict DESC,obstacle
        """
        with self._connection() as db:
            return [dict(row) for row in db.execute(query, (photo_id,))]

    def add_region(self, photo_id: int, box) -> int:
        """Register a human or model candidate using normalized xyxy coordinates."""
        try:
            x1, y1, x2, y2 = [round(float(v), 6) for v in box]
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Supply four normalized region coordinates.") from None
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(
                "The region must have positive area and lie inside the image."
            )
        params = (photo_id, x1, y1, x2, y2)
        with self._connection() as db:
            image_row = db.execute(
                "SELECT image FROM photos WHERE id=?", (photo_id,)
            ).fetchone()
            if image_row is None:
                raise KeyError(photo_id)
            with Image.open(BytesIO(image_row[0])) as img:
                if round(x2 * img.width) <= round(x1 * img.width) or round(
                    y2 * img.height
                ) <= round(y1 * img.height):
                    raise ValueError(
                        "The region must include at least one image pixel."
                    )
            db.execute(
                "INSERT INTO regions(photo_id,x1,y1,x2,y2) VALUES(?,?,?,?,?) "
                "ON CONFLICT(photo_id,x1,y1,x2,y2) DO NOTHING",
                params,
            )
            return db.execute(
                "SELECT id FROM regions WHERE photo_id=? AND x1=? "
                "AND y1=? AND x2=? AND y2=?",
                params,
            ).fetchone()[0]

    def regions(self, photo_id: int) -> list[dict]:
        return [
            {"id": r["id"], "box": r["box"]} for r in self.region_summaries(photo_id)
        ]

    def report_region(self, region_id: int, reviewer: str, verdict: str) -> None:
        reviewer = _text(reviewer, "Reviewer", 80).casefold()
        if verdict not in ("present", "absent", "uncertain"):
            raise ValueError("Choose present, absent, or uncertain.")
        with self._connection() as db:
            try:
                db.execute(
                    """INSERT INTO regional_reports(region_id,reviewer,verdict) VALUES(?,?,?)
                       ON CONFLICT(region_id,reviewer) DO UPDATE SET verdict=excluded.verdict""",
                    (region_id, reviewer, verdict),
                )
            except sqlite3.IntegrityError as exc:
                if exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY:
                    raise KeyError(region_id) from exc
                raise

    def region_summaries(self, photo_id: int) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM regional_evidence WHERE photo_id=? ORDER BY id",
                (photo_id,),
            )
            result = []
            for row in rows:
                yes, no, unclear = row["present"], row["absent"], row["uncertain"]
                if yes and no:
                    status = "conflicting"
                elif yes:
                    status = "reported present"
                elif no:
                    status = "reported absent"
                else:
                    status = "uncertain" if unclear else "unreviewed"
                result.append(
                    {
                        "id": row["id"],
                        "box": [row[k] for k in ("x1", "y1", "x2", "y2")],
                        "present": yes,
                        "absent": no,
                        "uncertain": unclear,
                        "conflict": bool(yes and no),
                        "status": status,
                    }
                )
            return result
