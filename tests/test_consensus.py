from plate_app.consensus import ConsensusTracker, intersection_over_union
from plate_app.history import HistoryStore


def detection(text="KL07AB1234", box=None, recognition_confidence=0.9):
    return {"id": "temporary", "text": text, "raw_text": text, "box": box or [0, 0, 100, 30],
            "detection_confidence": 0.9, "recognition_confidence": recognition_confidence, "status": "reading"}


def test_three_matching_readings_among_last_five():
    tracker = ConsensusTracker()
    readings = ["KL07AB1234", "KL07AB1235", "KL07AB1234", "KL07AB1236", "KL07AB1234"]
    for frame, text in enumerate(readings):
        result = tracker.update("camera", [detection(text)], now=float(frame))[0]
        assert result["status"] == ("confirmed" if frame == 4 else "reading")
    assert result["text"] == "KL07AB1234"


def test_two_plates_do_not_share_votes():
    tracker = ConsensusTracker()
    for frame in range(2):
        tracker.update("camera", [detection(), detection("AB123456", [200, 0, 300, 30])], now=float(frame))
    result = tracker.update("camera", [detection("AB123456", [200, 0, 300, 30]), detection("KL07AB1235")], now=2.0)
    assert result[0]["status"] == "confirmed"
    assert result[1]["status"] == "reading"
    assert result[0]["id"] != result[1]["id"]


def test_repeated_low_confidence_does_not_become_confirmed():
    tracker = ConsensusTracker()
    for frame in range(6):
        result = tracker.update("camera", [detection(recognition_confidence=0.4)], now=float(frame))
        assert result[0]["status"] == "needs_review"


def test_tracks_expire_and_sessions_are_separate():
    tracker = ConsensusTracker()
    for frame in range(3):
        previous = tracker.update("camera", [detection()], now=float(frame))[0]
    assert previous["status"] == "confirmed"
    assert tracker.update("other-camera", [detection()], now=3.0)[0]["status"] == "reading"
    expired = tracker.update("camera", [detection()], now=10.0)[0]
    assert expired["status"] == "reading" and expired["id"] != previous["id"]


def test_nonoverlapping_new_vehicle_starts_new_track():
    tracker = ConsensusTracker()
    for frame in range(3):
        tracker.update("camera", [detection()], now=float(frame))
    result = tracker.update("camera", [detection(box=[300, 100, 400, 130])], now=3.0)[0]
    assert result["status"] == "reading"


def test_history_dedup_expires_after_thirty_seconds(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    reading = detection()
    reading["status"] = "confirmed"
    first = store.record(reading, "live", "session", now=100.0)
    assert store.record(reading, "live", "session", now=120.0) == first
    assert store.record(reading, "live", "session", now=151.0) != first
    assert store.record(reading, "live", "different-session", now=151.0) != first
    assert len(store.list()) == 3


def test_iou_handles_zero_area():
    assert intersection_over_union([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
    assert intersection_over_union([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
