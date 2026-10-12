"""Small spatial tracker and conservative agreement for live camera readings."""

from collections import Counter, deque
from dataclasses import dataclass, field
from statistics import mean
import threading
import time
import uuid


def intersection_over_union(first: list[int], second: list[int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def is_reliable(text: str, detector_score: float, recognizer_score: float) -> bool:
    return 4 <= len(text) <= 12 and detector_score >= 0.5 and recognizer_score >= 0.70


@dataclass
class Track:
    box: list[int]
    last_seen: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    observations: deque = field(default_factory=lambda: deque(maxlen=5))
    misses: int = 0


class ConsensusTracker:
    def __init__(self):
        self._sessions: dict[str, list[Track]] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def update(self, session_id: str, detections: list[dict], now: float | None = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        with self._lock:
            # Limit abandoned sessions and tracks so an open browser cannot grow memory forever.
            self._sessions = {key: [track for track in tracks if now - track.last_seen <= 5 and track.misses < 2]
                              for key, tracks in self._sessions.items()}
            self._sessions = {key: tracks for key, tracks in self._sessions.items() if tracks}
            if session_id not in self._sessions and len(self._sessions) >= 128:
                oldest = min(self._sessions, key=lambda key: max(track.last_seen for track in self._sessions[key]))
                del self._sessions[oldest]
            tracks = self._sessions.setdefault(session_id, [])
            used = set()
            for detection in sorted(detections, key=lambda item: item["detection_confidence"], reverse=True):
                candidates = [(intersection_over_union(track.box, detection["box"]), track)
                              for track in tracks if track.id not in used]
                overlap, track = max(candidates, key=lambda pair: pair[0]) if candidates else (0.0, None)
                if overlap < 0.25:
                    track = Track(list(detection["box"]), now)
                    tracks.append(track)
                used.add(track.id)
                track.box = list(detection["box"])
                track.last_seen = now
                track.misses = 0
                text = detection["text"]
                reliable = is_reliable(text, detection["detection_confidence"], detection["recognition_confidence"])
                track.observations.append((text if reliable else "", detection["recognition_confidence"]))
                counts = Counter(value for value, _ in track.observations if value)
                winner, count = counts.most_common(1)[0] if counts else ("", 0)
                detection["id"] = track.id
                if count >= 3 and reliable:
                    detection["text"] = winner
                    detection["recognition_confidence"] = round(mean(score for value, score in track.observations if value == winner), 4)
                    detection["status"] = "confirmed"
                else:
                    detection["status"] = "reading" if reliable else "needs_review"
            for track in tracks:
                if track.id not in used:
                    track.misses += 1
                    # Missing detections break agreement gradually rather than preserving old votes.
                    track.observations.append(("", 0.0))
            self._sessions[session_id] = tracks[-64:]
        return detections
