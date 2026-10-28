import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from datasets import delete_uploaded_dataset, get_uploaded_dataset
from detection import MODEL_PATH, pig_detection_service

logger = logging.getLogger(__name__)
MAX_TRAINING_EPOCHS = 50
DEFAULT_TRAINING_EPOCHS = 20
IMAGE_SIZE = 640
TRAINING_ROOT = Path(__file__).resolve().parent / "training-runs"


class DatasetTrainingError(ValueError):
    pass


class DatasetTrainingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "idle",
            "dataset_id": None,
            "dataset_name": None,
            "epoch": 0,
            "epochs": DEFAULT_TRAINING_EPOCHS,
            "message": "No training job has been started.",
            "error": None,
        }
        self._process: subprocess.Popen[bytes] | None = None
        self._worker_status_path: Path | None = None
        self._candidate_path: Path | None = None
        self._run_directory: Path | None = None
        self._log_path: Path | None = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_worker_status()
            return dict(self._state)

    def _refresh_worker_status(self) -> None:
        if self._worker_status_path is None or not self._worker_status_path.is_file():
            return
        try:
            worker_status = json.loads(self._worker_status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read model training worker status.")
            return
        self._state.update(
            epoch=worker_status.get("epoch", self._state["epoch"]),
            epochs=worker_status.get("epochs", self._state["epochs"]),
            message=worker_status.get("message", self._state["message"]),
            error=worker_status.get("error") or self._state["error"],
        )

    def delete_dataset(self, dataset_id: str) -> None:
        with self._lock:
            if self._state["status"] == "running" and self._state["dataset_id"] == dataset_id:
                raise DatasetTrainingError("This dataset cannot be deleted while its training job is running.")
            delete_uploaded_dataset(dataset_id)

    def start(self, dataset_id: str, epochs: int = DEFAULT_TRAINING_EPOCHS) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "running":
                raise DatasetTrainingError("A dataset training job is already running.")
            if not 1 <= epochs <= MAX_TRAINING_EPOCHS:
                raise DatasetTrainingError(f"Training epochs must be between 1 and {MAX_TRAINING_EPOCHS}.")
            dataset_root = get_uploaded_dataset(dataset_id)
            self._dataset_config(dataset_root)
            self._state = {
                "status": "running",
                "dataset_id": dataset_id,
                "dataset_name": None,
                "epoch": 0,
                "epochs": epochs,
                "message": "Validating dataset and starting YOLO training.",
                "error": None,
            }
            TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
            run_directory = Path(tempfile.mkdtemp(prefix=f"{dataset_id}-", dir=TRAINING_ROOT))
            worker_status_path = run_directory / "worker-status.json"
            candidate_path = MODEL_PATH.with_name(
                f".{MODEL_PATH.stem}.{uuid.uuid4().hex}.tmp{MODEL_PATH.suffix}"
            )
            log_path = TRAINING_ROOT / f"{run_directory.name}.log"
            worker_command = [
                sys.executable,
                str(Path(__file__).with_name("training_worker.py")),
                "--dataset-id",
                dataset_id,
                "--epochs",
                str(epochs),
                "--run-directory",
                str(run_directory),
                "--status-path",
                str(worker_status_path),
                "--candidate-path",
                str(candidate_path),
            ]
            creation_flags = 0
            process_options: dict[str, Any] = {}
            if os.name == "nt":
                creation_flags = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
            else:
                process_options["start_new_session"] = True
            try:
                with log_path.open("ab") as log_file:
                    self._process = subprocess.Popen(
                        worker_command,
                        cwd=Path(__file__).resolve().parent,
                        stdin=subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        creationflags=creation_flags,
                        **process_options,
                    )
            except OSError as exc:
                shutil.rmtree(run_directory, ignore_errors=True)
                self._state.update(
                    status="failed",
                    message="Could not start the isolated model training worker.",
                    error=str(exc),
                )
                raise DatasetTrainingError("Could not start the model training worker.") from exc
            self._worker_status_path = worker_status_path
            self._candidate_path = candidate_path
            self._run_directory = run_directory
            self._log_path = log_path
            thread = threading.Thread(
                target=self._monitor_worker,
                name=f"pig-dataset-training-monitor-{uuid.uuid4().hex[:8]}",
                daemon=True,
            )
            thread.start()
            return dict(self._state)

    def _set_state(self, **updates: Any) -> None:
        with self._lock:
            self._state.update(updates)

    @staticmethod
    def _dataset_config(dataset_root: Path) -> tuple[Path, Path]:
        configs = sorted(
            (
                path
                for path in dataset_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            ),
            key=lambda path: path.as_posix().lower(),
        )
        if len(configs) > 1:
            config_names = ", ".join(path.relative_to(dataset_root).as_posix() for path in configs[:5])
            raise DatasetTrainingError(
                f"Found multiple YAML files in the uploaded dataset ({config_names}). Keep one YOLO data.yaml, "
                "or upload a Roboflow COCO export with train/valid/_annotations.coco.json files."
            )
        if not configs:
            coco_splits = DatasetTrainingService._coco_splits(dataset_root)
            if coco_splits is not None:
                source_config = coco_splits["splits"]["train"][0] / "_annotations.coco.json"
                return source_config, {
                    "__smartswine_format": "coco",
                    "names": coco_splits["names"],
                }
            raise DatasetTrainingError(
                "No YOLO data.yaml or Roboflow COCO annotations were found. Include one YOLO data.yaml, "
                "or a Roboflow COCO export with train/_annotations.coco.json and valid "
                "(or val)/_annotations.coco.json alongside the images."
            )

        source_config = configs[0]
        try:
            raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise DatasetTrainingError("Could not read the dataset's YOLO YAML configuration.") from exc
        if not isinstance(raw, dict):
            raise DatasetTrainingError("The YOLO data YAML must contain a mapping.")

        names = raw.get("names")
        if isinstance(names, dict):
            class_names = list(names.values())
        elif isinstance(names, list):
            class_names = names
        else:
            raise DatasetTrainingError("The YOLO data YAML must define class names as a list or mapping.")
        if not class_names or any(not isinstance(name, str) or not name.strip() for name in class_names):
            raise DatasetTrainingError("The YOLO data YAML contains invalid class names.")
        if raw.get("nc", len(class_names)) != len(class_names):
            raise DatasetTrainingError("The class count does not match the YOLO data YAML names.")

        dataset_base = source_config.parent
        configured_base = raw.get("path", ".")
        if not isinstance(configured_base, str):
            raise DatasetTrainingError("The YOLO data YAML path must be a local folder path.")
        base_path = Path(configured_base)
        if base_path.is_absolute():
            resolved_base = base_path.resolve()
        else:
            resolved_base = (dataset_base / base_path).resolve()
        try:
            resolved_base.relative_to(dataset_root)
        except ValueError as exc:
            raise DatasetTrainingError("The YOLO data YAML path must stay inside the uploaded dataset.") from exc
        if not resolved_base.is_dir():
            raise DatasetTrainingError("The dataset path configured in the YOLO data YAML does not exist.")

        normalized = dict(raw)
        normalized["path"] = str(resolved_base)
        for split in ("train", "val", "test"):
            if split not in raw:
                continue
            sources = raw[split]
            source_list = sources if isinstance(sources, list) else [sources]
            resolved_sources = []
            for source in source_list:
                if not isinstance(source, str):
                    raise DatasetTrainingError(f"The YOLO data YAML {split} entries must be paths.")
                candidate = Path(source)
                resolved_source = (resolved_base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
                try:
                    resolved_source.relative_to(dataset_root)
                except ValueError as exc:
                    raise DatasetTrainingError(
                        f"The YOLO data YAML {split} paths must stay inside the uploaded dataset."
                    ) from exc
                if not resolved_source.exists():
                    raise DatasetTrainingError(
                        f"The YOLO data YAML {split} path does not exist: {source}"
                    )
                resolved_sources.append(str(resolved_source))
            normalized[split] = resolved_sources if isinstance(sources, list) else resolved_sources[0]

        if "train" not in normalized or "val" not in normalized:
            raise DatasetTrainingError("The YOLO data YAML must define existing train and val paths.")
        return source_config, normalized

    @staticmethod
    def _coco_splits(dataset_root: Path) -> dict[str, Any] | None:
        annotations: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        for annotation_path in dataset_root.rglob("_annotations.coco.json"):
            split_name = annotation_path.parent.name.lower()
            normalized_split = "val" if split_name in {"valid", "validation"} else split_name
            if normalized_split in annotations:
                annotations[normalized_split].append(annotation_path)

        if not annotations["train"] and not annotations["val"]:
            return None
        if len(annotations["train"]) != 1 or len(annotations["val"]) != 1:
            raise DatasetTrainingError(
                "Roboflow COCO training requires exactly one train/_annotations.coco.json and one "
                "valid/_annotations.coco.json (or val/_annotations.coco.json)."
            )

        splits: dict[str, tuple[Path, dict[str, Any]]] = {}
        class_names: list[str] | None = None
        for split in ("train", "val", "test"):
            if len(annotations[split]) > 1:
                raise DatasetTrainingError(f"The Roboflow COCO dataset has more than one {split} annotation file.")
            if not annotations[split]:
                continue
            annotation_path = annotations[split][0]
            try:
                data = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetTrainingError(
                    f"Could not read the COCO annotations in {annotation_path.relative_to(dataset_root)}."
                ) from exc
            if not isinstance(data, dict):
                raise DatasetTrainingError("COCO annotation files must contain a JSON object.")
            categories = data.get("categories")
            images = data.get("images")
            records = data.get("annotations")
            if not isinstance(categories, list) or not categories or not isinstance(images, list) or not images or not isinstance(records, list):
                raise DatasetTrainingError("COCO annotations must define non-empty categories and images, and an annotations list.")
            current_names: list[str] = []
            category_ids: set[int] = set()
            for category in sorted(categories, key=lambda item: item.get("id", -1) if isinstance(item, dict) else -1):
                if not isinstance(category, dict) or not isinstance(category.get("id"), int):
                    raise DatasetTrainingError("A COCO category is missing a valid integer ID.")
                name = category.get("name")
                if not isinstance(name, str) or not name.strip() or category["id"] in category_ids:
                    raise DatasetTrainingError("COCO categories must have unique integer IDs and non-empty names.")
                category_ids.add(category["id"])
                current_names.append(name.strip())
            if class_names is None:
                class_names = current_names
            elif class_names != current_names:
                raise DatasetTrainingError("COCO train, validation, and test categories do not match.")
            splits[split] = (annotation_path.parent, data)

        return {"splits": splits, "names": class_names}

    @staticmethod
    def _prepare_coco_dataset(
        dataset_root: Path,
        coco_info: dict[str, Any],
        output_directory: Path,
    ) -> dict[str, Any]:
        image_root = output_directory / "images"
        label_root = output_directory / "labels"
        image_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
        names = coco_info["names"]

        for split, (source_directory, data) in coco_info["splits"].items():
            output_split_images = image_root / split
            output_split_labels = label_root / split
            output_split_images.mkdir(parents=True, exist_ok=True)
            output_split_labels.mkdir(parents=True, exist_ok=True)
            image_records: dict[int, dict[str, Any]] = {}
            categories = sorted(data["categories"], key=lambda category: category["id"])
            class_indices = {category["id"]: index for index, category in enumerate(categories)}
            for image in data["images"]:
                if not isinstance(image, dict) or not isinstance(image.get("id"), int):
                    raise DatasetTrainingError("A COCO image record is missing a valid integer ID.")
                if image["id"] in image_records:
                    raise DatasetTrainingError(f"The COCO {split} annotations contain duplicate image IDs.")
                filename = image.get("file_name")
                if not isinstance(filename, str) or not filename:
                    raise DatasetTrainingError(f"A COCO {split} image is missing its file name.")
                relative_filename = Path(filename)
                if relative_filename.is_absolute() or ".." in relative_filename.parts or "\\" in filename:
                    raise DatasetTrainingError("COCO image paths must be relative and stay within the dataset.")
                source_image = (source_directory / relative_filename).resolve()
                try:
                    source_image.relative_to(dataset_root)
                except ValueError as exc:
                    raise DatasetTrainingError("COCO image paths must stay inside the uploaded dataset.") from exc
                if not source_image.is_file():
                    raise DatasetTrainingError(f"COCO image file was not found: {filename}")
                if not isinstance(image.get("width"), int) or image["width"] <= 0 or not isinstance(image.get("height"), int) or image["height"] <= 0:
                    raise DatasetTrainingError(f"COCO image dimensions are invalid for {filename}.")
                image_records[image["id"]] = {
                    "width": image["width"],
                    "height": image["height"],
                    "filename": relative_filename,
                    "labels": [],
                }

            for annotation in data["annotations"]:
                if not isinstance(annotation, dict) or annotation.get("image_id") not in image_records:
                    raise DatasetTrainingError(f"A COCO {split} annotation refers to an unknown image.")
                category_id = annotation.get("category_id")
                if category_id not in class_indices:
                    raise DatasetTrainingError(f"A COCO {split} annotation refers to an unknown category.")
                bbox = annotation.get("bbox")
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox)
                ):
                    raise DatasetTrainingError("COCO bounding boxes must contain four finite numbers.")
                x, y, width, height = map(float, bbox)
                image_record = image_records[annotation["image_id"]]
                image_width = image_record["width"]
                image_height = image_record["height"]
                x1 = min(max(x, 0.0), float(image_width))
                y1 = min(max(y, 0.0), float(image_height))
                x2 = min(max(x + width, 0.0), float(image_width))
                y2 = min(max(y + height, 0.0), float(image_height))
                if x2 <= x1 or y2 <= y1:
                    continue
                center_x = ((x1 + x2) / 2) / image_width
                center_y = ((y1 + y2) / 2) / image_height
                box_width = (x2 - x1) / image_width
                box_height = (y2 - y1) / image_height
                class_index = class_indices[category_id]
                image_record["labels"].append(
                    f"{class_index} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                )

            for image_record in image_records.values():
                relative_filename = image_record["filename"]
                image_target = output_split_images / relative_filename
                label_target = output_split_labels / relative_filename.with_suffix(".txt")
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_directory / relative_filename, image_target)
                except OSError:
                    shutil.copy2(source_directory / relative_filename, image_target)
                label_target.write_text(
                    "\n".join(image_record["labels"]),
                    encoding="utf-8",
                )

        config: dict[str, Any] = {
            "path": str(output_directory),
            "train": "images/train",
            "val": "images/val",
            "names": names,
            "nc": len(names),
        }
        if "test" in coco_info["splits"]:
            config["test"] = "images/test"
        return config

    def _monitor_worker(self) -> None:
        with self._lock:
            process = self._process
            run_directory = self._run_directory
            status_path = self._worker_status_path
            candidate_path = self._candidate_path
            log_path = self._log_path
        if process is None or run_directory is None or status_path is None or candidate_path is None:
            return
        try:
            return_code = process.wait()
            self._refresh_worker_status()
            if return_code != 0:
                error = self._state.get("error")
                if not error:
                    error = f"The training worker exited unexpectedly with code {return_code}."
                raise DatasetTrainingError(error)
            if not candidate_path.is_file():
                raise DatasetTrainingError("The training worker finished without producing new detector weights.")

            from ultralytics import YOLO

            trained_model = YOLO(str(candidate_path))
            os.replace(candidate_path, MODEL_PATH)
            pig_detection_service.activate_model(trained_model)
            self._set_state(
                status="completed",
                epoch=self._state["epochs"],
                message="Training complete. The new detector is active in live monitoring.",
                error=None,
            )
        except Exception as exc:
            if not isinstance(exc, DatasetTrainingError):
                logger.exception("Could not activate trained detector weights.")
            self._set_state(
                status="failed",
                message="Dataset training failed. Check the backend logs for details.",
                error=str(exc) or exc.__class__.__name__,
            )
        finally:
            candidate_path.unlink(missing_ok=True)
            shutil.rmtree(run_directory, ignore_errors=True)
            if log_path is not None and self._state["status"] == "completed":
                log_path.unlink(missing_ok=True)
            with self._lock:
                self._process = None
                self._worker_status_path = None
                self._candidate_path = None
                self._run_directory = None
                self._log_path = None

    @staticmethod
    def _read_metadata(dataset_root: Path) -> dict[str, Any]:
        from datasets import UPLOAD_MARKER

        return json.loads((dataset_root / UPLOAD_MARKER).read_text(encoding="utf-8"))


def _write_worker_status(
    status_path: Path,
    *,
    epoch: int,
    epochs: int,
    message: str,
    error: str | None = None,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {"epoch": epoch, "epochs": epochs, "message": message, "error": error},
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, status_path)


def run_training_worker(
    dataset_id: str,
    epochs: int,
    run_directory: Path,
    status_path: Path,
    candidate_path: Path,
) -> int:
    run_directory = run_directory.resolve()
    try:
        dataset_root = get_uploaded_dataset(dataset_id)
        metadata = DatasetTrainingService._read_metadata(dataset_root)
        source_config, normalized_config = DatasetTrainingService._dataset_config(dataset_root)
        if normalized_config.get("__smartswine_format") == "coco":
            coco_info = DatasetTrainingService._coco_splits(dataset_root)
            if coco_info is None:
                raise DatasetTrainingError("The Roboflow COCO annotations could not be found.")
            normalized_config = DatasetTrainingService._prepare_coco_dataset(
                dataset_root,
                coco_info,
                run_directory / "coco-yolo",
            )
        normalized_yaml = run_directory / "dataset.yaml"
        normalized_yaml.write_text(
            yaml.safe_dump(normalized_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        _write_worker_status(
            status_path,
            epoch=0,
            epochs=epochs,
            message=f"Training {metadata.get('name', dataset_id)} with {source_config.name}.",
        )

        from ultralytics import YOLO

        if not MODEL_PATH.is_file():
            raise DatasetTrainingError(
                "The current YOLO detector weights are missing; restore pig_detector.pt before training."
            )
        model = YOLO(str(MODEL_PATH))

        def record_epoch(trainer: Any) -> None:
            epoch = int(trainer.epoch) + 1
            total_epochs = int(trainer.epochs)
            _write_worker_status(
                status_path,
                epoch=epoch,
                epochs=total_epochs,
                message=f"Training epoch {epoch} of {total_epochs}.",
            )

        model.add_callback("on_train_epoch_end", record_epoch)
        model.train(
            data=str(normalized_yaml),
            epochs=epochs,
            imgsz=IMAGE_SIZE,
            project=str(run_directory),
            name="weights",
            exist_ok=True,
            verbose=False,
            plots=False,
            batch=4,
            workers=0,
            amp=False,
        )
        best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best_weights.is_file():
            raise DatasetTrainingError("YOLO training completed without producing best.pt weights.")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, candidate_path)
        _write_worker_status(
            status_path,
            epoch=epochs,
            epochs=epochs,
            message="Training completed. Activating the best model weights.",
        )
        return 0
    except Exception as exc:
        logger.exception("Dataset training worker failed.")
        _write_worker_status(
            status_path,
            epoch=0,
            epochs=epochs,
            message="Dataset training failed. Check the backend training log.",
            error=str(exc) or exc.__class__.__name__,
        )
        return 1


dataset_training_service = DatasetTrainingService()
