import io
import json
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import datasets


class DatasetUploadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root.mkdir(exist_ok=True)
        self.root_patch = patch.object(datasets, "DATASET_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.directory.cleanup)

    @staticmethod
    def archive(entries):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            for name, data in entries:
                archive.writestr(name, data)
        content.seek(0)
        return content

    def test_extracts_dataset_and_lists_upload(self):
        archive = self.archive([
            ("pig-dataset/data.yaml", "path: .\n"),
            ("pig-dataset/images/train/image.jpg", b"image"),
        ])

        uploaded = datasets.save_dataset_archive(archive, "pig-dataset.zip")

        self.assertEqual(uploaded["name"], "pig-dataset.zip")
        self.assertEqual(uploaded["file_count"], 2)
        self.assertEqual((self.root / uploaded["folder"] / "pig-dataset" / "data.yaml").read_text(), "path: .\n")
        self.assertEqual(datasets.list_uploaded_datasets(), [uploaded])
        marker = json.loads((self.root / uploaded["folder"] / datasets.UPLOAD_MARKER).read_text())
        self.assertEqual(marker, uploaded)

    def test_rejects_path_traversal_without_writing_outside_dataset_root(self):
        outside = self.root.parent / f"smartswine-test-{uuid.uuid4().hex}.txt"
        archive = self.archive([("../outside.txt", "unsafe")])

        with self.assertRaises(datasets.DatasetArchiveError):
            datasets.save_dataset_archive(archive, "unsafe.zip")

        self.assertFalse(outside.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_windows_special_paths(self):
        for filename in ("dataset/image.jpg:stream", "dataset/NUL.txt", "dataset/trailing. "):
            with self.subTest(filename=filename):
                with self.assertRaises(datasets.DatasetArchiveError):
                    datasets.save_dataset_archive(self.archive([(filename, "unsafe")]), "unsafe.zip")
                self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_non_zip_archive(self):
        with self.assertRaises(datasets.DatasetArchiveError):
            datasets.save_dataset_archive(io.BytesIO(b"not a zip"), "dataset.zip")

    def test_rejects_file_and_child_path_conflict(self):
        archive = self.archive([("dataset", "file"), ("dataset/image.jpg", "child")])

        with self.assertRaises(datasets.DatasetArchiveError):
            datasets.save_dataset_archive(archive, "dataset.zip")

        self.assertEqual(list(self.root.iterdir()), [])

    def test_deletes_only_uploaded_dataset_directories(self):
        uploaded = datasets.save_dataset_archive(
            self.archive([("dataset/data.yaml", "names: [pig]\n")]),
            "dataset.zip",
        )

        datasets.delete_uploaded_dataset(uploaded["id"])

        self.assertEqual(datasets.list_uploaded_datasets(), [])
        with self.assertRaises(datasets.DatasetNotFoundError):
            datasets.delete_uploaded_dataset(uploaded["id"])


if __name__ == "__main__":
    unittest.main()
