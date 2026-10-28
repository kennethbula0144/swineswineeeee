import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from label_posture_review import (
    DEFAULT_CLASSES,
    PostureLabelerError,
    _read_review_manifest,
    write_labeled_review_zip,
)


class PostureLabelerTests(unittest.TestCase):
    def test_default_posture_labels_are_distinct_primary_postures(self):
        self.assertEqual(
            DEFAULT_CLASSES,
            ("standing", "sitting", "sternal_lying", "lateral_lying"),
        )

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source_zip = self.root / "review.zip"
        self.output_zip = self.root / "review-labeled.zip"
        with zipfile.ZipFile(self.source_zip, "w") as archive:
            archive.writestr(
                "manifest.csv",
                "crop_file,posture_label,pig_identity_from_source,source_split,source_image\n"
                "to_label/crop.jpg,,Pig 1,train,frame.jpg\n",
            )
            archive.writestr("to_label/crop.jpg", b"crop")

    def test_reads_crop_manifest_and_exports_user_labels(self):
        with zipfile.ZipFile(self.source_zip) as archive:
            manifest_name, rows = _read_review_manifest(archive)

        self.assertEqual(manifest_name, "manifest.csv")
        self.assertEqual(rows[0]["posture_label"], "")
        rows[0]["posture_label"] = "standing"
        write_labeled_review_zip(self.source_zip, self.output_zip, manifest_name, rows)

        with zipfile.ZipFile(self.output_zip) as archive:
            exported_manifest = archive.read("manifest.csv").decode("utf-8-sig")
            self.assertIn("to_label/crop.jpg,standing,Pig 1,train,frame.jpg", exported_manifest)
            self.assertEqual(archive.read("to_label/crop.jpg"), b"crop")

    def test_rejects_manifest_crop_path_traversal(self):
        with zipfile.ZipFile(self.source_zip, "w") as archive:
            archive.writestr(
                "manifest.csv",
                "crop_file,posture_label,source_image\n../outside.jpg,,frame.jpg\n",
            )
            archive.writestr("../outside.jpg", b"unsafe")

        with zipfile.ZipFile(self.source_zip) as archive:
            with self.assertRaisesRegex(PostureLabelerError, "Unsafe crop_file"):
                _read_review_manifest(archive)

    def test_refuses_to_overwrite_original_review_zip(self):
        with zipfile.ZipFile(self.source_zip) as archive:
            manifest_name, rows = _read_review_manifest(archive)

        with self.assertRaisesRegex(PostureLabelerError, "must not overwrite"):
            write_labeled_review_zip(self.source_zip, self.source_zip, manifest_name, rows)


if __name__ == "__main__":
    unittest.main()
