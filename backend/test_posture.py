import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import httpx

from app import app
from posture import PostureDetectionService
from posture_training import PostureTrainingError, PostureTrainingService


class PostureDetectionServiceTests(unittest.TestCase):
    def test_reports_unconfigured_model_without_failing_pig_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PostureDetectionService(Path(directory) / "missing.pt")
            result = service.classify_detections(
                np.zeros((100, 100, 3), dtype=np.uint8),
                [{"class_name": "pig", "box": {"x": 10, "y": 10, "w": 50, "h": 50}}],
            )

        self.assertFalse(result["posture_status"]["available"])
        self.assertIn("PIG_POSTURE_MODEL", result["posture_status"]["message"])
        self.assertNotIn("posture", result["detections"][0])

    def test_classifies_each_detection_crop_using_model_class_names(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "posture.pt"
            model_path.touch()
            service = PostureDetectionService(model_path)
            probabilities = SimpleNamespace(top1=1, top1conf=SimpleNamespace(item=lambda: 0.93))
            crop_shapes = []

            def predict(source, verbose):
                crop_shapes.append(source.shape[:2])
                return [SimpleNamespace(probs=probabilities)]

            model = SimpleNamespace(
                names={0: "standing", 1: "sternal_lying"},
                predict=predict,
            )
            with patch.object(service, "_load_model", return_value=model):
                result = service.classify_detections(
                    np.zeros((100, 100, 3), dtype=np.uint8),
                    [{"class_name": "pig", "box": {"x": 10, "y": 10, "w": 50, "h": 50}}],
                )

        self.assertTrue(result["posture_status"]["available"])
        self.assertEqual(result["posture_status"]["classes"], ["standing", "sternal_lying"])
        self.assertEqual(crop_shapes, [(50, 50)])
        self.assertEqual(
            result["detections"][0]["posture"],
            {"label": "sternal_lying", "confidence": 0.93},
        )

    def test_skips_posture_classification_for_unknown_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "posture.pt"
            model_path.touch()
            service = PostureDetectionService(model_path)
            predict = unittest.mock.Mock(
                return_value=[
                    SimpleNamespace(
                        probs=SimpleNamespace(
                            top1=0,
                            top1conf=SimpleNamespace(item=lambda: 0.9),
                        )
                    )
                ]
            )
            model = SimpleNamespace(names={0: "standing"}, predict=predict)
            detections = [
                {"class_name": "Unknown", "box": {"x": 10, "y": 10, "w": 20, "h": 20}},
                {"class_name": "Pig 1", "box": {"x": 40, "y": 40, "w": 20, "h": 20}},
            ]
            with patch.object(service, "_load_model", return_value=model):
                result = service.classify_detections(
                    np.zeros((100, 100, 3), dtype=np.uint8),
                    detections,
                )

        self.assertEqual(predict.call_count, 1)
        self.assertNotIn("posture", result["detections"][0])
        self.assertEqual(result["detections"][1]["posture"]["label"], "standing")

    def test_reports_incompatible_detector_model_without_losing_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "posture.pt"
            model_path.touch()
            service = PostureDetectionService(model_path)
            model = SimpleNamespace(names={0: "standing"}, predict=lambda source, verbose: [SimpleNamespace(probs=None)])
            detection = {"class_name": "pig", "box": {"x": 10, "y": 10, "w": 50, "h": 50}}
            with patch.object(service, "_load_model", return_value=model):
                result = service.classify_detections(
                    np.zeros((100, 100, 3), dtype=np.uint8),
                    [detection],
                )

        self.assertFalse(result["posture_status"]["available"])
        self.assertIn("YOLO image-classification", result["posture_status"]["message"])
        self.assertEqual(result["detections"], [detection])

    def test_posture_training_accepts_labeled_train_and_validation_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                for posture in ("standing", "lateral_lying"):
                    image_directory = root / split / posture
                    image_directory.mkdir(parents=True)
                    (image_directory / "sample.jpg").touch()

            self.assertEqual(
                PostureTrainingService._classification_root(root),
                root,
            )

    def test_posture_training_accepts_validation_split_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation"):
                for posture in ("standing", "lateral_lying"):
                    image_directory = root / split / posture
                    image_directory.mkdir(parents=True)
                    (image_directory / "sample.jpg").touch()

            self.assertEqual(PostureTrainingService._classification_root(root), root)

    def test_posture_training_accepts_roboflow_valid_split_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "valid"):
                for posture in ("standing", "lateral_lying"):
                    image_directory = root / split / posture
                    image_directory.mkdir(parents=True)
                    self.assertTrue(
                        cv2.imwrite(
                            str(image_directory / "sample.jpg"),
                            np.zeros((16, 16, 3), dtype=np.uint8),
                        )
                    )

            classification_root = PostureTrainingService._classification_root(root)
            with patch("posture_training.MODEL_PATH", root / "models" / "pig_posture.pt"):
                prepared, cleanup_root = PostureTrainingService._prepare_classification_dataset(
                    classification_root
                )
            self.assertIsNotNone(cleanup_root)
            self.addCleanup(lambda: shutil.rmtree(cleanup_root, ignore_errors=True))

            self.assertTrue((prepared / "val" / "standing" / "sample.jpg").is_file())
            self.assertTrue((prepared / "val" / "lateral_lying" / "sample.jpg").is_file())
            self.assertTrue((root / "valid" / "standing" / "sample.jpg").is_file())

    def test_posture_training_finds_classification_dataset_nested_in_zip_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "outer" / "export" / "dataset"
            for split in ("train", "val"):
                for posture in ("standing", "lateral_lying"):
                    image_directory = nested / split / posture
                    image_directory.mkdir(parents=True)
                    (image_directory / "sample.jpg").touch()

            self.assertEqual(PostureTrainingService._classification_root(root), nested)

    def test_posture_training_converts_roboflow_coco_posture_boxes_to_image_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            categories = [
                {"id": 1, "name": "lying"},
                {"id": 2, "name": "sitting"},
                {"id": 3, "name": "standing"},
                {"id": 4, "name": "uncertain"},
                {"id": 5, "name": "Pig 1"},
            ]
            for split in ("train", "valid"):
                split_root = root / split
                split_root.mkdir()
                labels = ["lying", "standing", "uncertain", "Pig 1"]
                if split == "train":
                    labels.append("sitting")
                images = []
                annotations = []
                for index, label in enumerate(labels):
                    filename = f"{label.replace(' ', '-')}-{index}.jpg"
                    self.assertTrue(
                        cv2.imwrite(
                            str(split_root / filename),
                            np.full((32, 32, 3), index, dtype=np.uint8),
                        )
                    )
                    images.append(
                        {"id": index, "file_name": filename, "width": 32, "height": 32}
                    )
                    category_id = next(
                        category["id"] for category in categories if category["name"] == label
                    )
                    annotations.append(
                        {
                            "id": index,
                            "image_id": index,
                            "category_id": category_id,
                            "bbox": [4, 4, 20, 20],
                        }
                    )
                (split_root / "_annotations.coco.json").write_text(
                    json.dumps(
                        {
                            "categories": categories,
                            "images": images,
                            "annotations": annotations,
                        }
                    ),
                    encoding="utf-8",
                )

            with patch("posture_training.MODEL_PATH", root / "models" / "pig_posture.pt"):
                prepared = PostureTrainingService._prepare_coco_posture_dataset(root)

            self.assertIsNotNone(prepared)
            self.addCleanup(lambda: shutil.rmtree(prepared, ignore_errors=True))
            for split in ("train", "val"):
                self.assertTrue(list((prepared / split / "lying").glob("*.jpg")))
                self.assertTrue(list((prepared / split / "standing").glob("*.jpg")))
                self.assertFalse((prepared / split / "sitting").exists())
                self.assertFalse((prepared / split / "uncertain").exists())
                self.assertFalse((prepared / split / "pig_1").exists())

    def test_posture_training_does_not_treat_pig_identity_coco_categories_as_postures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "valid"):
                split_root = root / split
                split_root.mkdir()
                (split_root / "_annotations.coco.json").write_text(
                    json.dumps(
                        {
                            "categories": [
                                {"id": 1, "name": "Pig 1"},
                                {"id": 2, "name": "Pig 2"},
                            ],
                            "images": [],
                            "annotations": [],
                        }
                    ),
                    encoding="utf-8",
                )

            with patch("posture_training.MODEL_PATH", root / "models" / "pig_posture.pt"):
                with self.assertRaisesRegex(PostureTrainingError, "at least two posture classes"):
                    PostureTrainingService._prepare_coco_posture_dataset(root)

    def test_posture_training_prepares_labeled_review_manifest_without_frame_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_root = root / "review"
            crops = review_root / "to_label"
            crops.mkdir(parents=True)
            manifest_path = review_root / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.DictWriter(
                    manifest_file,
                    fieldnames=("crop_file", "posture_label", "source_image"),
                )
                writer.writeheader()
                for index in range(20):
                    source_frame = f"frame-{index}.jpg"
                    for posture in ("standing", "lateral_lying"):
                        crop_name = f"{index}-{posture}.jpg"
                        image_path = crops / crop_name
                        self.assertTrue(cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8)))
                        writer.writerow(
                            {
                                "crop_file": f"to_label/{crop_name}",
                                "posture_label": posture,
                                "source_image": source_frame,
                            }
                        )

            with patch("posture_training.MODEL_PATH", root / "models" / "pig_posture.pt"):
                prepared = PostureTrainingService._prepare_review_manifest(root)

            self.assertIsNotNone(prepared)
            self.addCleanup(lambda: shutil.rmtree(prepared, ignore_errors=True))
            for split in ("train", "val"):
                for posture in ("standing", "lateral_lying"):
                    self.assertTrue(list((prepared / split / posture).glob("*.jpg")))
            for source_index in range(20):
                source_frame = f"frame-{source_index}.jpg"
                expected_split = (
                    "val"
                    if int.from_bytes(
                        hashlib.sha256(source_frame.encode("utf-8")).digest()[:8],
                        byteorder="big",
                    ) % 5 == 0
                    else "train"
                )
                for posture in ("standing", "lateral_lying"):
                    source_crop = crops / f"{source_index}-{posture}.jpg"
                    destination_name = hashlib.sha256(
                        (source_frame + str(source_crop)).encode("utf-8")
                    ).hexdigest()[:24] + ".jpg"
                    self.assertTrue(
                        (prepared / expected_split / posture / destination_name).is_file()
                    )
            self.assertTrue(list((prepared / "train").rglob("*.jpg")))
            self.assertTrue(list((prepared / "val").rglob("*.jpg")))

    def test_posture_training_accepts_label_first_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "labels.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as label_file:
                writer = csv.writer(label_file)
                writer.writerow(("label", "image_path", "split", "source_image"))
                for posture in ("standing", "lateral_lying"):
                    for index, split in enumerate(("train", "train", "valid", "valid")):
                        image_name = f"{posture}-{index}.jpg"
                        self.assertTrue(
                            cv2.imwrite(
                                str(root / image_name),
                                np.zeros((16, 16, 3), dtype=np.uint8),
                            )
                        )
                        writer.writerow((posture, image_name, split, image_name))

            with patch("posture_training.MODEL_PATH", root / "models" / "pig_posture.pt"):
                prepared = PostureTrainingService._prepare_review_manifest(root)

            self.assertIsNotNone(prepared)
            self.addCleanup(lambda: shutil.rmtree(prepared, ignore_errors=True))
            for split in ("train", "val"):
                for posture in ("standing", "lateral_lying"):
                    self.assertEqual(len(list((prepared / split / posture).glob("*.jpg"))), 2)

    def test_posture_training_reports_unlabeled_review_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_root = root / "review"
            review_root.mkdir()
            (review_root / "manifest.csv").write_text(
                "crop_file,posture_label,source_image\n"
                "to_label/one.jpg,,frame-1.jpg\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PostureTrainingError, "fill the posture_label column"):
                PostureTrainingService._prepare_review_manifest(root)

    def test_posture_training_rejects_detection_datasets_without_posture_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            (root / "val").mkdir()

            with self.assertRaisesRegex(PostureTrainingError, "at least two labeled posture classes"):
                PostureTrainingService._classification_root(root)

    def test_posture_training_activates_the_best_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "models" / "pig_posture.pt"
            output.parent.mkdir()
            training_root = root / "dataset"
            (training_root / "train" / "standing").mkdir(parents=True)
            (training_root / "val" / "standing").mkdir(parents=True)
            (training_root / "train" / "lateral_lying").mkdir()
            (training_root / "val" / "lateral_lying").mkdir()
            service = PostureTrainingService()
            service._state["status"] = "running"
            classifier = object()

            class FakeYOLO:
                def __init__(self, model_path):
                    self.model_path = model_path

                def add_callback(self, _name, callback):
                    self.callback = callback

                def train(self, **options):
                    run_directory = Path(options["project"]) / options["name"]
                    weights = run_directory / "weights"
                    weights.mkdir(parents=True)
                    (weights / "best.pt").touch()
                    self.callback(SimpleNamespace(epoch=0, epochs=1))

            ultralytics = ModuleType("ultralytics")
            ultralytics.YOLO = FakeYOLO
            with (
                patch("posture_training.MODEL_PATH", output),
                patch("posture_training.posture_detection_service.activate_model") as activate,
                patch.dict("sys.modules", {"ultralytics": ultralytics}),
            ):
                service._train(training_root, 1)

            self.assertEqual(service.status()["status"], "completed")
            self.assertTrue(output.is_file())
            activate.assert_called_once()


class PostureTrainingApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_posture_training_status_endpoint_is_registered(self):
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45683))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/posture-training/status")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("status", response.json())


if __name__ == "__main__":
    unittest.main()
