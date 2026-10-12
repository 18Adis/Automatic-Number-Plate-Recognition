"""Geometry, leakage, malformed annotation and evaluation regression coverage."""
import math
import unittest

from scripts.dataset_utils import clipped_bbox, grouped_split, plate_crop, plate_issues, safe_member, source_group, yolo_bbox
from scripts.evaluate import text_metrics


class DatasetTests(unittest.TestCase):
    def test_roboflow_variants_share_source(self):
        self.assertEqual(source_group({"file_name": "V_P_I_12_jpg.rf.abc.jpg"}), "v_p_i_12")
        self.assertEqual(source_group({"file_name": "random.jpg", "extra": {"name": "V_P_I_12.jpg"}}), "v_p_i_12")

    def test_group_and_byte_duplicate_leakage(self):
        records = [{"member": f"x/{i}_{j}.jpg", "source_group": f"g{i}", "sha256": f"hash{i}_{j}"}
                   for i in range(40) for j in range(3)]
        # Different named sources may still be exactly identical byte-for-byte.
        records[0]["sha256"] = records[-1]["sha256"]
        result = grouped_split(records, seed=42)
        again = grouped_split(list(reversed(records)), seed=42)
        self.assertEqual(result, again)
        self.assertEqual(result[records[0]["member"]], result[records[-1]["member"]])
        for group in {r["source_group"] for r in records}:
            self.assertEqual(len({result[r["member"]][0] for r in records if r["source_group"] == group}), 1)
        self.assertEqual({s for s, _ in result.values()}, {"train", "val", "test"})

    def test_clipping_and_crop_normalization(self):
        self.assertEqual(clipped_bbox([-10, 10, 30, 20], 100, 100), (0, 10, 20, 30))
        self.assertEqual(yolo_bbox((110, 220, 130, 240), 100, 50, origin=(100, 200)), (.2, .6, .2, .4))
        self.assertEqual(yolo_bbox((-10, 10, 20, 30), 100, 100), (.1, .2, .2, .2))

    def test_invalid_boxes_rejected(self):
        for box in ([0, 0, 0, 3], [0, 0, -2, 3], [0, 0, math.nan, 2], [200, 0, 1, 2], [1, 2, 3], None):
            with self.subTest(box=box), self.assertRaises(ValueError):
                clipped_bbox(box, 100, 100)

    def test_crop_encloses_clipped_characters(self):
        crop = plate_crop([(1, 2, 10, 20), (20, 3, 30, 20)], 32, 24)
        self.assertEqual(crop, (0, 0, 32, 24))
        self.assertTrue(all(0 <= v <= 1 for v in yolo_bbox((1, 2, 10, 20), 32, 24, crop[:2])))

    def test_unsafe_paths_rejected(self):
        for path in ("../evil.jpg", "/absolute.jpg", "C:/evil.jpg", "a\\..\\evil.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_member(path)

    def test_disconnected_characters_flag_review(self):
        boxes = [(i * 10, 10, i * 10 + 7, 25) for i in range(5)]
        self.assertEqual(plate_issues(boxes), [])
        self.assertIn("disconnected_character_groups", plate_issues(boxes + [(400, 10, 407, 25)]))

    def test_missing_plate_read_counts_as_failure(self):
        metrics = text_metrics({"a": "KL01AA1234", "b": "AB12"}, {"a": "KL01AA1234"})
        self.assertEqual(metrics["exact_registration_accuracy"], .5)
        self.assertEqual(metrics["missing_predictions"], 1)
        self.assertEqual(metrics["character_edits"], 4)


if __name__ == "__main__":
    unittest.main()
