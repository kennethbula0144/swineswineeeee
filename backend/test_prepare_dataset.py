import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from prepare_dataset import DatasetPreparationError, DatasetPreparer


class DatasetPreparerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "prepared.zip"

    @staticmethod
    def write_image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), np.zeros((32, 48, 3), dtype=np.uint8))
        if not success:
            raise AssertionError(f"Could not create test image {path}")

    def test_converts_coco_valid_split_and_skips_missing_images(self):
        for split in ("train", "valid"):
            self.write_image(self.source / split / "pig.jpg")
            document = {
                "categories": [{"id": 7, "name": "Pig 1"}],
                "images": [
                    {"id": 1, "file_name": "pig.jpg", "width": 48, "height": 32},
                    {"id": 2, "file_name": "missing.jpg", "width": 48, "height": 32},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 7, "bbox": [-4, 4, 20, 12]},
                    {"image_id": 1, "category_id": 7, "bbox": ["bad", 0, 1, 1]},
                    {"image_id": 2, "category_id": 7, "bbox": [0, 0, 10, 10]},
                ],
            }
            (self.source / split / "_annotations.coco.json").write_text(
                json.dumps(document),
                encoding="utf-8",
            )

        result = DatasetPreparer(self.source, self.output).prepare("auto")

        self.assertEqual(result["format"], "coco")
        self.assertTrue(any("missing or corrupt COCO image" in warning for warning in result["warnings"]))
        with zipfile.ZipFile(self.output) as archive:
            config = yaml.safe_load(archive.read("data.yaml"))
            self.assertEqual(config["val"], "images/val")
            labels = archive.read("labels/train/pig.txt").decode("utf-8").splitlines()
            self.assertEqual(len(labels), 1)
            self.assertTrue(archive.getinfo("images/val/pig.jpg"))

    def test_normalizes_yolo_paths_and_cleans_invalid_labels(self):
        for split in ("train", "valid"):
            self.write_image(self.source / "images" / split / "pig.jpg")
            labels = self.source / "labels" / split / "pig.txt"
            labels.parent.mkdir(parents=True, exist_ok=True)
            labels.write_text("0 0.5 0.5 0.4 0.3\nbad label\n", encoding="utf-8")
        (self.source / "data.yaml").write_text(
            "path: old-machine/dataset\ntrain: images/train\nvalid: images/valid\nnames: [pig]\n",
            encoding="utf-8",
        )

        result = DatasetPreparer(self.source, self.output).prepare("auto")

        self.assertEqual(result["format"], "yolo")
        self.assertTrue(any("Removed invalid label" in warning for warning in result["warnings"]))
        with zipfile.ZipFile(self.output) as archive:
            config = yaml.safe_load(archive.read("data.yaml"))
            self.assertEqual(config["path"], ".")
            self.assertEqual(config["val"], "images/val")
            self.assertEqual(
                archive.read("labels/train/pig.txt").decode("utf-8"),
                "0 0.500000 0.500000 0.400000 0.300000",
            )

    def test_normalizes_posture_validation_split_and_skips_corrupt_images(self):
        for split in ("train", "validation"):
            for class_name in ("standing", "lateral_lying"):
                self.write_image(self.source / split / class_name / "pig.jpg")
                corrupt = self.source / split / class_name / "broken.jpg"
                corrupt.write_bytes(b"not an image")

        result = DatasetPreparer(self.source, self.output).prepare("auto")

        self.assertEqual(result["format"], "posture")
        self.assertEqual(len(result["warnings"]), 4)
        with zipfile.ZipFile(self.output) as archive:
            self.assertIn("train/standing/pig.jpg", archive.namelist())
            self.assertIn("val/lateral_lying/pig.jpg", archive.namelist())
            self.assertNotIn("val/standing/broken.jpg", archive.namelist())

    def test_rejects_a_prepared_dataset_over_the_uncompressed_limit(self):
        for split in ("train", "val"):
            for class_name in ("standing", "lateral_lying"):
                self.write_image(self.source / split / class_name / "pig.jpg")

        with patch("prepare_dataset.MAX_EXTRACTED_BYTES", 1):
            with self.assertRaisesRegex(DatasetPreparationError, "Uncompressed dataset exceeds"):
                DatasetPreparer(self.source, self.output).prepare("posture")
        self.assertFalse(self.output.exists())

    def test_rejects_compressed_zip_over_limit_without_replacing_output(self):
        for split in ("train", "val"):
            for class_name in ("standing", "lateral_lying"):
                self.write_image(self.source / split / class_name / "pig.jpg")
        self.output.write_bytes(b"previous file")

        with patch("prepare_dataset.MAX_ARCHIVE_BYTES", 1):
            with self.assertRaisesRegex(DatasetPreparationError, "Compressed ZIP"):
                DatasetPreparer(self.source, self.output).prepare("posture")

        self.assertEqual(self.output.read_bytes(), b"previous file")


if __name__ == "__main__":
    unittest.main()
