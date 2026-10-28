import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from extract_posture_review import PostureReviewError, create_review_bundle


class PostureReviewBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.output = self.root / "review.zip"

    def _make_coco_split(self, split: str) -> None:
        split_directory = self.source / split
        split_directory.mkdir(parents=True)
        image = np.full((80, 100, 3), (20, 100, 200), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(split_directory / "frame.jpg"), image))
        annotations = {
            "categories": [
                {"id": 1, "name": "Pig 1"},
                {"id": 2, "name": "batch"},
            ],
            "images": [
                {"id": 10, "file_name": "frame.jpg"},
                {"id": 11, "file_name": "missing.jpg"},
            ],
            "annotations": [
                {"id": 100, "image_id": 10, "category_id": 1, "bbox": [10, 12, 30, 20]},
                {"id": 101, "image_id": 10, "category_id": 2, "bbox": [0, 0, 90, 70]},
                {"id": 102, "image_id": 11, "category_id": 1, "bbox": [0, 0, 20, 20]},
            ],
        }
        (split_directory / "_annotations.coco.json").write_text(
            json.dumps(annotations),
            encoding="utf-8",
        )

    def test_extracts_only_identity_crops_and_leaves_posture_unlabeled(self):
        self.source.mkdir()
        self._make_coco_split("train")
        self._make_coco_split("valid")

        result = create_review_bundle(self.source, self.output)

        self.assertEqual(result["crops"], 2)
        with zipfile.ZipFile(self.output) as archive:
            manifest = archive.read("manifest.csv").decode("utf-8-sig")
            self.assertIn("posture_label", manifest)
            self.assertIn(",Pig 1,train,frame.jpg,100", manifest)
            self.assertIn(",Pig 1,val,frame.jpg,100", manifest)
            self.assertNotIn("batch", manifest)
            self.assertIn("to_label/train_00001.jpg", archive.namelist())
            self.assertIn("to_label/val_00002.jpg", archive.namelist())
            image_data = np.frombuffer(archive.read("to_label/train_00001.jpg"), dtype=np.uint8)
            crop = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            self.assertEqual(crop.shape[:2], (20, 30))

    def test_rejects_when_no_individual_pig_boxes_exist(self):
        self.source.mkdir()
        self._make_coco_split("train")
        self._make_coco_split("valid")
        for annotation_path in self.source.rglob("_annotations.coco.json"):
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            data["annotations"] = [item for item in data["annotations"] if item["category_id"] == 2]
            annotation_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaisesRegex(PostureReviewError, "No valid pig identity crops"):
            create_review_bundle(self.source, self.output)

    def test_rejects_bundle_over_compressed_size_limit_without_replacing_output(self):
        self.source.mkdir()
        self._make_coco_split("train")
        self._make_coco_split("valid")
        self.output.write_bytes(b"previous output")

        with patch("extract_posture_review.MAX_ARCHIVE_BYTES", 1):
            with self.assertRaisesRegex(PostureReviewError, "512 MB compressed"):
                create_review_bundle(self.source, self.output)

        self.assertEqual(self.output.read_bytes(), b"previous output")


if __name__ == "__main__":
    unittest.main()
