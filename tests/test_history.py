import sqlite3

from plate_app.history import HistoryStore


DETECTION = {"text": "KL07AB1234", "raw_text": "KL 07 AB 1234", "detection_confidence": 0.92,
             "recognition_confidence": 0.88, "status": "confirmed"}


def test_old_database_migration_preserves_history_and_handles_missing_images(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE detections (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, raw_text TEXT NOT NULL,
            source TEXT NOT NULL, timestamp TEXT NOT NULL,
            detection_confidence REAL NOT NULL, recognition_confidence REAL NOT NULL,
            status TEXT NOT NULL, session_id TEXT, last_seen REAL NOT NULL)""")
        connection.execute("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?)",
                           ("a" * 32, "KL07AB1234", "KL 07 AB 1234", "upload", "2026-10-12T00:00:00Z", 0.92, 0.88, "confirmed", None, 0))
    store = HistoryStore(path)
    item = store.list()[0]
    assert item["text"] == "KL07AB1234"
    assert item["image_url"] is None and item["thumbnail_url"] is None
    assert store.capture(item["id"]) is None


def test_dedup_keeps_first_timestamp_capture_and_correction_keeps_image(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    identity = store.record(DETECTION, "live", "camera", now=1000, image_jpeg=b"frame-one", crop_jpeg=b"crop-one")
    assert store.record(DETECTION, "live", "camera", now=1005, image_jpeg=b"frame-two", crop_jpeg=b"crop-two") == identity
    assert len(store.list()) == 1
    assert store.capture(identity) == b"frame-one"
    assert store.capture(identity, thumbnail=True) == b"crop-one"
    assert store.list()[0]["timestamp"].startswith("1970-01-01T00:16:40")
    store.correct(identity, "KL07AB1235")
    assert store.capture(identity) == b"frame-one"
    assert store.list()[0]["text"] == "KL07AB1235"
