import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import training
from datasets import UPLOAD_MARKER


class DatasetTrainingConfigTests(unittest.TestCase):
    def test_detects_and_converts_roboflow_coco_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "uploaded"
            source_root = root / "roboflow-export"
            categories = [
                {"id": 0, "name": "pig"},
                {"id": 1, "name": "batch"},
            ]
            for split in ("train", "valid", "test"):
                split_directory = source_root / split
                split_directory.mkdir(parents=True)
                (split_directory / "sample.jpg").write_bytes(b"image")
                (split_directory / "_annotations.coco.json").write_text(
                    json.dumps({
                        "categories": categories,
                        "images": [{"id": 4, "file_name": "sample.jpg", "width": 200, "height": 100}],
                        "annotations": [
                            {"id": 1, "image_id": 4, "category_id": 1, "bbox": [20, 10, 40, 20]},
                            {"id": 2, "image_id": 4, "category_id": 0, "bbox": [100, 40, 50, 30]},
                        ],
                    }),
                    encoding="utf-8",
                )

            source, config = training.DatasetTrainingService._dataset_config(root)
            output = Path(directory) / "yolo-view"
            converted = training.DatasetTrainingService._prepare_coco_dataset(
                root,
                training.DatasetTrainingService._coco_splits(root),
                output,
            )

            self.assertTrue(source.name.endswith("_annotations.coco.json"))
            self.assertEqual(config["__smartswine_format"], "coco")
            self.assertEqual(converted["names"], ["pig", "batch"])
            self.assertEqual(converted["train"], "images/train")
            self.assertEqual(converted["val"], "images/val")
            self.assertEqual(converted["test"], "images/test")
            self.assertEqual(
                (output / "labels" / "train" / "sample.txt").read_text(encoding="utf-8").splitlines(),
                ["1 0.200000 0.200000 0.200000 0.200000", "0 0.625000 0.550000 0.250000 0.300000"],
            )
            self.assertTrue((output / "images" / "train" / "sample.jpg").is_file())

    def test_explains_required_coco_train_and_validation_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "uploaded"
            train = root / "train"
            train.mkdir(parents=True)
            (train / "_annotations.coco.json").write_text(
                json.dumps({
                    "categories": [{"id": 1, "name": "pig"}],
                    "images": [{"id": 1, "file_name": "missing.jpg", "width": 10, "height": 10}],
                    "annotations": [],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(training.DatasetTrainingError, "train/.*valid"):
                training.DatasetTrainingService._dataset_config(root)

    def test_normalizes_paths_for_yolo_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "uploaded"
            yaml_directory = root / "dataset"
            (yaml_directory / "images" / "train").mkdir(parents=True)
            (yaml_directory / "images" / "val").mkdir(parents=True)
            config = yaml_directory / "data.yaml"
            config.write_text(
                "train: images/train\nval: images/val\nnames:\n  0: pig\n",
                encoding="utf-8",
            )

            source, normalized = training.DatasetTrainingService._dataset_config(root)

            self.assertEqual(source, config)
            self.assertEqual(normalized["path"], str(yaml_directory.resolve()))
            self.assertEqual(normalized["train"], str((yaml_directory / "images" / "train").resolve()))
            self.assertEqual(normalized["val"], str((yaml_directory / "images" / "val").resolve()))
            self.assertEqual(normalized["names"], {0: "pig"})

    def test_rejects_dataset_yaml_paths_outside_uploaded_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "uploaded"
            yaml_directory = root / "dataset"
            yaml_directory.mkdir(parents=True)
            outside = Path(directory) / "outside"
            outside.mkdir()
            (yaml_directory / "data.yaml").write_text(
                f"path: {outside}\ntrain: .\nval: .\nnames: [pig]\n",
                encoding="utf-8",
            )

            with self.assertRaises(training.DatasetTrainingError):
                training.DatasetTrainingService._dataset_config(root)

    def test_start_rejects_invalid_dataset_without_marking_training_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_id = "dataset-123"
            dataset_directory = root / dataset_id
            dataset_directory.mkdir()
            (dataset_directory / UPLOAD_MARKER).write_text(
                json.dumps({"id": dataset_id, "folder": dataset_id, "name": "dataset.zip"}),
                encoding="utf-8",
            )
            with patch("datasets.DATASET_ROOT", root):
                service = training.DatasetTrainingService()
                with self.assertRaises(training.DatasetTrainingError):
                    service.start(dataset_id)

            self.assertEqual(service.status()["status"], "idle")

    def test_cannot_delete_dataset_while_it_is_being_trained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_id = "dataset-123"
            dataset_directory = root / dataset_id
            dataset_directory.mkdir()
            (dataset_directory / UPLOAD_MARKER).write_text(
                json.dumps({"id": dataset_id, "folder": dataset_id}),
                encoding="utf-8",
            )
            service = training.DatasetTrainingService()
            service._state.update(status="running", dataset_id=dataset_id)
            with patch("datasets.DATASET_ROOT", root):
                with self.assertRaises(training.DatasetTrainingError):
                    service.delete_dataset(dataset_id)

            self.assertTrue(dataset_directory.is_dir())

    def test_completed_training_atomically_activates_best_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_id = "dataset-123"
            dataset_directory = root / "datasets" / dataset_id
            images_directory = dataset_directory / "images"
            (images_directory / "train").mkdir(parents=True)
            (images_directory / "val").mkdir(parents=True)
            (dataset_directory / "data.yaml").write_text(
                "train: images/train\nval: images/val\nnames: [pig]\n",
                encoding="utf-8",
            )
            (dataset_directory / UPLOAD_MARKER).write_text(
                json.dumps({"id": dataset_id, "folder": dataset_id, "name": "dataset.zip"}),
                encoding="utf-8",
            )
            configured_model = root / "pig_detector.pt"
            configured_model.write_bytes(b"old weights")

            class FakeYOLO:
                def __init__(self, _weights):
                    self.callbacks = []

                def add_callback(self, _event, callback):
                    self.callbacks.append(callback)

                def train(self, **options):
                    output = Path(options["project"]) / options["name"]
                    weights = output / "weights" / "best.pt"
                    weights.parent.mkdir(parents=True)
                    weights.write_bytes(b"trained weights")
                    self.trainer = types.SimpleNamespace(save_dir=output)

                    class Trainer:
                        epoch = 0
                        epochs = 1

                    for callback in self.callbacks:
                        callback(Trainer())

            class FinishedProcess:
                @staticmethod
                def wait():
                    return 0

            service = training.DatasetTrainingService()
            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
            run_directory = root / "run"
            run_directory.mkdir()
            worker_status = run_directory / "worker-status.json"
            candidate_weights = root / "candidate.pt"
            log_path = root / "training.log"
            with (
                patch("datasets.DATASET_ROOT", root / "datasets"),
                patch.object(training, "MODEL_PATH", configured_model),
                patch.object(training.pig_detection_service, "activate_model") as activate,
                patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
            ):
                result = training.run_training_worker(
                    dataset_id,
                    1,
                    run_directory,
                    worker_status,
                    candidate_weights,
                )
                self.assertEqual(result, 0)
                service._state.update(status="running", dataset_id=dataset_id, epochs=1)
                service._process = FinishedProcess()
                service._worker_status_path = worker_status
                service._candidate_path = candidate_weights
                service._run_directory = run_directory
                service._log_path = log_path
                log_path.write_text("training output", encoding="utf-8")
                service._monitor_worker()

            self.assertEqual(service.status()["status"], "completed")
            self.assertEqual(configured_model.read_bytes(), b"trained weights")
            activate.assert_called_once()
            self.assertFalse(log_path.exists())


if __name__ == "__main__":
    unittest.main()
