"""Project-local detection history with JPEG captures for image review."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
import time
import uuid


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, raw_text TEXT NOT NULL,
                source TEXT NOT NULL, timestamp TEXT NOT NULL,
                detection_confidence REAL NOT NULL, recognition_confidence REAL NOT NULL,
                status TEXT NOT NULL, session_id TEXT, last_seen REAL NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS history_recent ON detections(session_id, text, last_seen)")
            connection.execute("""CREATE TABLE IF NOT EXISTS captures (
                detection_id TEXT PRIMARY KEY REFERENCES detections(id) ON DELETE CASCADE,
                image_jpeg BLOB NOT NULL, crop_jpeg BLOB NOT NULL
            )""")

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def record(self, detection: dict, source: str, session_id: str | None = None, now: float | None = None,
               image_jpeg: bytes | None = None, crop_jpeg: bytes | None = None) -> str:
        now = time.time() if now is None else now
        with self._lock, self._connect() as connection:
            if source in {"live", "video"} and session_id:
                previous = connection.execute(
                    "SELECT id FROM detections WHERE session_id=? AND text=? AND source=? AND last_seen>=? ORDER BY last_seen DESC LIMIT 1",
                    (session_id, detection["text"], source, now - 30),
                ).fetchone()
                if previous:
                    connection.execute("UPDATE detections SET last_seen=? WHERE id=?", (now, previous["id"]))
                    self._save_capture(connection, previous["id"], image_jpeg, crop_jpeg)
                    return previous["id"]
            identity = uuid.uuid4().hex
            timestamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
            connection.execute(
                "INSERT INTO detections (id,text,raw_text,source,timestamp,detection_confidence,recognition_confidence,status,session_id,last_seen) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (identity, detection["text"], detection["raw_text"], source, timestamp,
                 detection["detection_confidence"], detection["recognition_confidence"],
                 detection["status"], session_id, now),
            )
            self._save_capture(connection, identity, image_jpeg, crop_jpeg)
            return identity

    @staticmethod
    def _save_capture(connection, identity, image_jpeg, crop_jpeg):
        if image_jpeg and crop_jpeg:
            # Preserve the frame belonging to the original timestamp, including on live dedup.
            connection.execute("INSERT OR IGNORE INTO captures VALUES (?,?,?)", (identity, image_jpeg, crop_jpeg))

    def capture(self, identity: str, thumbnail: bool = False) -> bytes | None:
        column = "crop_jpeg" if thumbnail else "image_jpeg"
        with self._connect() as connection:
            row = connection.execute(f"SELECT {column} FROM captures WHERE detection_id=?", (identity,)).fetchone()
            return bytes(row[0]) if row else None

    def list(self, limit: int = 1000) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("""SELECT d.id,d.text,d.raw_text,d.source,d.timestamp,
                d.detection_confidence,d.recognition_confidence,d.status,
                CASE WHEN c.detection_id IS NULL THEN 0 ELSE 1 END AS has_image
                FROM detections d LEFT JOIN captures c ON c.detection_id=d.id
                ORDER BY d.timestamp DESC LIMIT ?""", (limit,))
            result = []
            for row in rows:
                item = dict(row)
                has_image = bool(item.pop("has_image"))
                item["image_url"] = f"/api/history/{item['id']}/image" if has_image else None
                item["thumbnail_url"] = f"/api/history/{item['id']}/thumbnail" if has_image else None
                result.append(item)
            return result

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM detections")

    def correct(self, identity: str, text: str) -> dict | None:
        with self._lock, self._connect() as connection:
            # A human correction is saved explicitly; raw_text keeps the original OCR result.
            result = connection.execute("UPDATE detections SET text=?, status='confirmed' WHERE id=?", (text, identity))
            if result.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT id,text,raw_text,source,timestamp,detection_confidence,recognition_confidence,status FROM detections WHERE id=?", (identity,)
            ).fetchone()
            return dict(row)
