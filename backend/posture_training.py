import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

from datasets import get_uploaded_dataset
from posture import MODEL_PATH, posture_detection_service

logger = logging.getLogger(__name__)
DEFAULT_POSTURE_EPOCHS = 20
MAX_POSTURE_EPOCHS = 50
IMAGE_SIZE = 224
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class PostureTrainingError(ValueError):
    pass


class PostureTrainingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "idle",
            "dataset_id": None,
            "dataset_name": None,
            "epoch": 0,
            "epochs": DEFAULT_POSTURE_EPOCHS,
            "message": "No posture training job has been started.",
            "error": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    @staticmethod
    def _classification_root(dataset_root: Path) -> Path:
        candidates = [
            directory
            for directory in (dataset_root, *dataset_root.rglob("*"))
            if directory.is_dir()
            and (directory / "train").is_dir()
            and any((directory / split).is_dir() for split in ("val", "valid", "validation"))
        ]
        if len(candidates) != 1:
            raise PostureTrainingError(
                "Posture training needs one image-classification dataset with "
                "train/<posture-name>/ and val/<posture-name>/ (or valid/ or validation/) folders."
            )

        root = candidates[0]
        validation_directory = next(
            root / split
            for split in ("val", "valid", "validation")
            if (root / split).is_dir()
        )
        train_classes = {
            directory.name
            for directory in (root / "train").iterdir()
            if directory.is_dir()
        }
        val_classes = {
            directory.name
            for directory in validation_directory.iterdir()
            if directory.is_dir()
        }
        if len(train_classes) < 2:
            raise PostureTrainingError(
                "Posture training needs at least two labeled posture classes in train/."
            )
        if train_classes != val_classes:
            raise PostureTrainingError(
                "The train/ and val/ folders must contain the same posture class folders."
            )
        for split_directory in (root / "train", validation_directory):
            for class_name in train_classes:
                class_directory = split_directory / class_name
                image_count = sum(
                    path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    for path in class_directory.rglob("*")
                )
                if image_count == 0:
                    raise PostureTrainingError(
                        f"The {split_directory.name}/{class_name} folder must contain labeled images."
                    )
        return root

    @staticmethod
    def _prepare_classification_dataset(dataset_root: Path) -> tuple[Path, Path | None]:
        if (dataset_root / "val").is_dir():
            return dataset_root, None

        validation_directory = next(
            dataset_root / split
            for split in ("valid", "validation")
            if (dataset_root / split).is_dir()
        )
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        prepared_root = Path(tempfile.mkdtemp(prefix=".posture-classification-data-", dir=MODEL_PATH.parent))
        try:
            shutil.copytree(dataset_root / "train", prepared_root / "train")
            shutil.copytree(validation_directory, prepared_root / "val")
            return prepared_root, prepared_root
        except Exception:
            shutil.rmtree(prepared_root, ignore_errors=True)
            raise

    @staticmethod
    def _prepare_coco_posture_dataset(dataset_root: Path) -> Path | None:
        annotations: dict[str, list[Path]] = {"train": [], "val": []}
        for annotation_path in dataset_root.rglob("_annotations.coco.json"):
            split = annotation_path.parent.name.casefold()
            if split in {"train", "val", "valid", "validation"}:
                annotations["train" if split == "train" else "val"].append(annotation_path)
        if not annotations["train"] and not annotations["val"]:
            return None
        if len(annotations["train"]) != 1 or len(annotations["val"]) != 1:
            raise PostureTrainingError(
                "Roboflow posture COCO datasets need exactly one train and one "
                "valid/val _annotations.coco.json file."
            )

        excluded_labels = {
            "background",
            "ignore",
            "other",
            "uncertain",
            "unknown",
        }
        excluded_identity_label = re.compile(r"pig[\s_-]*\d+", re.IGNORECASE)
        prepared_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        prepared_root = Path(tempfile.mkdtemp(prefix=".posture-coco-data-", dir=MODEL_PATH.parent))
        try:
            for split, annotation_paths in annotations.items():
                annotation_path = annotation_paths[0]
                try:
                    data = json.loads(annotation_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PostureTrainingError(
                        f"Could not read Roboflow COCO {split} annotations."
                    ) from exc
                if not isinstance(data, dict):
                    raise PostureTrainingError(f"Roboflow COCO {split} annotations must contain an object.")
                categories = data.get("categories")
                images = data.get("images")
                records = data.get("annotations")
                if not isinstance(categories, list) or not isinstance(images, list) or not isinstance(records, list):
                    raise PostureTrainingError(
                        f"Roboflow COCO {split} annotations need categories, images, and annotations lists."
                    )
                category_labels: dict[Any, str] = {}
                for category in categories:
                    if not isinstance(category, dict) or not isinstance(category.get("name"), str):
                        continue
                    category_name = category["name"].strip().lower().replace(" ", "_")
                    if (
                        category_name
                        and category_name not in excluded_labels
                        and not excluded_identity_label.fullmatch(category_name)
                        and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,59}", category_name)
                    ):
                        category_labels[category.get("id")] = category_name

                image_records = {
                    image.get("id"): image
                    for image in images
                    if isinstance(image, dict) and image.get("id") is not None
                }
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        continue
                    label = category_labels.get(record.get("category_id"))
                    image_record = image_records.get(record.get("image_id"))
                    if label is None or image_record is None:
                        continue
                    image_name = image_record.get("file_name")
                    if not isinstance(image_name, str) or not image_name:
                        continue
                    relative_image = PurePosixPath(image_name)
                    if relative_image.is_absolute() or "\\" in image_name or ".." in relative_image.parts:
                        raise PostureTrainingError(
                            f"Unsafe image path in Roboflow COCO {split} annotations."
                        )
                    image_path = (annotation_path.parent / Path(*relative_image.parts)).resolve(strict=False)
                    try:
                        image_path.relative_to(dataset_root.resolve())
                    except ValueError as exc:
                        raise PostureTrainingError(
                            f"Image path in Roboflow COCO {split} annotations escapes the dataset."
                        ) from exc
                    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        logger.warning("Skipping missing or corrupt COCO posture image: %s", image_path)
                        continue
                    bbox = record.get("bbox")
                    if (
                        not isinstance(bbox, list)
                        or len(bbox) != 4
                        or any(
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(value)
                            for value in bbox
                        )
                    ):
                        logger.warning("Skipping invalid COCO posture bounding box in %s.", annotation_path)
                        continue
                    x, y, width, height = map(float, bbox)
                    image_height, image_width = image.shape[:2]
                    left = max(0, min(image_width, math.floor(x)))
                    top = max(0, min(image_height, math.floor(y)))
                    right = max(0, min(image_width, math.ceil(x + width)))
                    bottom = max(0, min(image_height, math.ceil(y + height)))
                    if right <= left or bottom <= top:
                        logger.warning("Skipping empty COCO posture crop in %s.", annotation_path)
                        continue
                    class_directory = prepared_root / split / label
                    class_directory.mkdir(parents=True, exist_ok=True)
                    annotation_id = record.get("id", index)
                    crop_name = hashlib.sha256(
                        f"{image_path}:{annotation_id}:{index}".encode("utf-8")
                    ).hexdigest()[:24] + ".jpg"
                    if cv2.imwrite(str(class_directory / crop_name), image[top:bottom, left:right]):
                        prepared_counts[split][label] = prepared_counts[split].get(label, 0) + 1
                    else:
                        raise PostureTrainingError("Could not write a prepared posture image crop.")

            usable_classes = {
                label
                for label, count in prepared_counts["train"].items()
                if count > 0 and prepared_counts["val"].get(label, 0) > 0
            }
            skipped_classes = (
                set(prepared_counts["train"])
                | set(prepared_counts["val"])
            ) - usable_classes
            if skipped_classes:
                logger.warning(
                    "Skipping COCO posture classes without images in both train and validation: %s",
                    ", ".join(sorted(skipped_classes)),
                )
                for split in ("train", "val"):
                    for label in skipped_classes:
                        shutil.rmtree(prepared_root / split / label, ignore_errors=True)
            if len(usable_classes) < 2:
                counts = ", ".join(
                    f"{label} (train={prepared_counts['train'].get(label, 0)}, "
                    f"val={prepared_counts['val'].get(label, 0)})"
                    for label in sorted(set(prepared_counts["train"]) | set(prepared_counts["val"]))
                ) or "no usable posture categories"
                raise PostureTrainingError(
                    "Roboflow COCO posture training needs at least two posture classes with annotated "
                    f"images in both train and valid splits. Found: {counts}. "
                    "Uncertain/unknown labels are ignored."
                )
            for split in ("train", "val"):
                for label in usable_classes:
                    if not any((prepared_root / split / label).iterdir()):
                        raise PostureTrainingError(
                            f"Roboflow COCO posture class '{label}' has no images in {split}."
                        )
            return prepared_root
        except Exception:
            shutil.rmtree(prepared_root, ignore_errors=True)
            raise

    @staticmethod
    def _prepare_label_first_manifest(
        dataset_root: Path,
        manifest_path: Path,
        fieldnames: list[str] | None,
        rows: list[dict[str, str | None]],
    ) -> Path:
        if not fieldnames or len(fieldnames) < 2:
            raise PostureTrainingError(
                "A label-first CSV needs a label column followed by an image path column."
            )
        label_header = (fieldnames[0] or "").strip().casefold()
        if label_header not in {"label", "posture", "posture_label", "class"}:
            raise PostureTrainingError(
                "CSV posture labels must be in the first column (label, posture, posture_label, or class)."
            )

        normalized_headers = [(name or "").strip().casefold() for name in fieldnames]
        image_index = next(
            (
                index
                for index, name in enumerate(normalized_headers[1:], 1)
                if name in {"image", "image_path", "filename", "file_name", "file", "path", "crop_file"}
            ),
            1,
        )
        split_index = next(
            (index for index, name in enumerate(normalized_headers) if name in {"split", "subset"}),
            None,
        )
        source_index = next(
            (
                index
                for index, name in enumerate(normalized_headers)
                if name in {"source_image", "source", "source_frame"}
            ),
            None,
        )

        labeled_rows: list[tuple[Path, str, str, str | None]] = []
        for line_number, row in enumerate(rows, 2):
            values = [row.get(name) or "" for name in fieldnames]
            label = values[0].strip().lower().replace(" ", "_")
            if not label:
                continue
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,59}", label):
                raise PostureTrainingError(
                    f"Invalid posture label on CSV line {line_number}: use letters, numbers, underscores, or hyphens."
                )
            image_name = values[image_index].strip()
            if not image_name:
                raise PostureTrainingError(f"CSV line {line_number} is missing its image path.")
            relative_image = Path(image_name.replace("\\", "/"))
            if relative_image.is_absolute() or ".." in relative_image.parts:
                raise PostureTrainingError(f"Unsafe image path on CSV line {line_number}.")
            candidates = (
                manifest_path.parent / relative_image,
                dataset_root / relative_image,
            )
            image_path = next((path.resolve() for path in candidates if path.is_file()), None)
            if image_path is None:
                raise PostureTrainingError(f"Could not find the image listed on CSV line {line_number}.")
            try:
                image_path.relative_to(dataset_root.resolve())
            except ValueError as exc:
                raise PostureTrainingError(f"Image path on CSV line {line_number} escapes the dataset.") from exc
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise PostureTrainingError(f"CSV line {line_number} does not point to an image.")
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                raise PostureTrainingError(f"The image on CSV line {line_number} is corrupt.")

            split = values[split_index].strip().casefold() if split_index is not None else ""
            if split in {"valid", "validation"}:
                split = "val"
            elif split not in {"train", "val"}:
                split = ""
            if not split:
                for part in image_path.relative_to(dataset_root.resolve()).parts:
                    normalized_part = part.casefold()
                    if normalized_part in {"train", "val", "valid", "validation"}:
                        split = "val" if normalized_part in {"val", "valid", "validation"} else "train"
                        break
            source_name = (
                values[source_index].strip().replace("\\", "/")
                if source_index is not None and values[source_index].strip()
                else image_path.relative_to(dataset_root.resolve()).as_posix()
            )
            labeled_rows.append((image_path, label, source_name, split or None))

        classes = {label for _, label, _, _ in labeled_rows}
        if len(classes) < 2:
            raise PostureTrainingError(
                "A label-first CSV needs at least two distinct posture labels and labeled images."
            )

        class_sources: dict[str, set[str]] = {label: set() for label in classes}
        for _, label, source_name, _ in labeled_rows:
            class_sources[label].add(source_name)
        for label, sources in class_sources.items():
            if len(sources) < 2:
                raise PostureTrainingError(
                    f"Posture class '{label}' needs at least two different images or source frames for train/validation."
                )

        source_split: dict[str, str] = {}
        explicit_sources: set[str] = set()
        for _, _, source_name, split in labeled_rows:
            if split and source_name in source_split and source_split[source_name] != split:
                raise PostureTrainingError(
                    f"CSV assigns source frame '{source_name}' to both training and validation."
                )
            if split:
                source_split[source_name] = split
                explicit_sources.add(source_name)
        for source_name in class_sources.values():
            for name in source_name:
                source_split.setdefault(
                    name,
                    "val"
                    if int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") % 5 == 0
                    else "train",
                )

        validation_sources = {name for name, split in source_split.items() if split == "val"}
        for label, sources in class_sources.items():
            if not any(source in validation_sources for source in sources):
                candidates = [source for source in sources if source not in explicit_sources]
                if not candidates:
                    raise PostureTrainingError(
                        f"Posture class '{label}' has no validation images in the CSV split."
                    )
                validation_sources.add(
                    min(candidates, key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
                )
            if all(source in validation_sources for source in sources):
                candidates = [source for source in sources if source not in explicit_sources]
                if not candidates:
                    raise PostureTrainingError(
                        f"Posture class '{label}' has no training images in the CSV split."
                    )
                validation_sources.remove(
                    max(candidates, key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
                )
        for label, sources in class_sources.items():
            if not any(source in validation_sources for source in sources) or all(
                source in validation_sources for source in sources
            ):
                raise PostureTrainingError(
                    f"Could not create both training and validation images for posture class '{label}'."
                )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        prepared_root = Path(tempfile.mkdtemp(prefix=".posture-csv-data-", dir=MODEL_PATH.parent))
        try:
            for image_path, label, source_name, _ in labeled_rows:
                split = "val" if source_name in validation_sources else "train"
                destination_directory = prepared_root / split / label
                destination_directory.mkdir(parents=True, exist_ok=True)
                destination_name = hashlib.sha256(
                    (source_name + str(image_path)).encode("utf-8")
                ).hexdigest()[:24] + image_path.suffix.lower()
                shutil.copy2(image_path, destination_directory / destination_name)
            for split in ("train", "val"):
                for label in classes:
                    if not any((prepared_root / split / label).iterdir()):
                        raise PostureTrainingError(
                            f"Posture class '{label}' needs images in both the train and validation splits."
                        )
            return prepared_root
        except Exception:
            shutil.rmtree(prepared_root, ignore_errors=True)
            raise

    @staticmethod
    def _prepare_review_manifest(dataset_root: Path) -> Path | None:
        manifests = [
            path
            for path in dataset_root.rglob("*.csv")
            if path.name.casefold() in {"manifest.csv", "labels.csv"}
        ]
        if not manifests:
            return None
        if len(manifests) != 1:
            raise PostureTrainingError(
                "The posture review dataset must contain exactly one manifest.csv."
            )
        manifest_path = manifests[0]
        try:
            with manifest_path.open(newline="", encoding="utf-8-sig") as manifest_file:
                reader = csv.DictReader(manifest_file)
                rows = list(reader)
                if not reader.fieldnames or not {
                    "crop_file",
                    "posture_label",
                    "source_image",
                }.issubset(reader.fieldnames):
                    return PostureTrainingService._prepare_label_first_manifest(
                        dataset_root,
                        manifest_path,
                        reader.fieldnames,
                        rows,
                    )
                required_columns = {"crop_file", "posture_label", "source_image"}
                if not required_columns.issubset(reader.fieldnames):
                    raise PostureTrainingError("manifest.csv is missing required posture-review columns.")
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise PostureTrainingError("Could not read the posture review manifest.csv.") from exc

        labeled_rows: list[tuple[Path, str, str]] = []
        unlabeled_count = 0
        for line_number, row in enumerate(rows, 2):
            label = (row.get("posture_label") or "").strip().lower().replace(" ", "_")
            if not label:
                unlabeled_count += 1
                continue
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,59}", label):
                raise PostureTrainingError(
                    f"Invalid posture label on manifest.csv line {line_number}: use letters, numbers, underscores, or hyphens."
                )
            crop_name = row.get("crop_file", "")
            source_name = row.get("source_image", "").strip()
            if not crop_name or not source_name:
                raise PostureTrainingError(
                    f"manifest.csv line {line_number} needs crop_file and source_image values."
                )
            crop_path = (manifest_path.parent / crop_name).resolve(strict=False)
            try:
                crop_path.relative_to(dataset_root.resolve())
            except ValueError as exc:
                raise PostureTrainingError(
                    f"Crop path on manifest.csv line {line_number} escapes the dataset."
                ) from exc
            if not crop_path.is_file() or crop_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise PostureTrainingError(
                    f"Could not find the image listed on manifest.csv line {line_number}."
                )
            image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if image is None:
                raise PostureTrainingError(
                    f"The crop image on manifest.csv line {line_number} is corrupt."
                )
            labeled_rows.append((crop_path, label, source_name.replace("\\", "/")))

        if unlabeled_count:
            logger.info("Ignoring %s unlabeled posture-review crops.", unlabeled_count)
        classes = {label for _, label, _ in labeled_rows}
        if len(classes) < 2:
            raise PostureTrainingError(
                f"Only {len(classes)} labeled posture class(es) found. Open the review ZIP's manifest.csv, "
                "fill the posture_label column with the actual label for each reviewed crop, save/re-ZIP, "
                "then upload the reviewed ZIP. At least two distinct postures are required; blank rows are skipped."
            )

        class_sources: dict[str, set[str]] = {label: set() for label in classes}
        source_labels: dict[str, set[str]] = {}
        for _, label, source_name in labeled_rows:
            class_sources[label].add(source_name)
            source_labels.setdefault(source_name, set()).add(label)
        for label, sources in class_sources.items():
            if len(sources) < 2:
                raise PostureTrainingError(
                    f"Posture class '{label}' needs images from at least two different source frames for train/validation."
                )

        source_split = {
            source_name: int.from_bytes(
                hashlib.sha256(source_name.encode("utf-8")).digest()[:8],
                byteorder="big",
            ) % 5 == 0
            for source_name in source_labels
        }
        validation_sources = {name for name, is_validation in source_split.items() if is_validation}
        for label, sources in class_sources.items():
            if not any(source in validation_sources for source in sources):
                chosen = min(sources, key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
                validation_sources.add(chosen)
            if all(source in validation_sources for source in sources):
                chosen = max(sources, key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
                validation_sources.remove(chosen)
        for label, sources in class_sources.items():
            if not any(source in validation_sources for source in sources) or all(
                source in validation_sources for source in sources
            ):
                raise PostureTrainingError(
                    f"Could not create both training and validation images for posture class '{label}'. "
                    "Add labeled crops from more source frames."
                )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        prepared_root = Path(tempfile.mkdtemp(prefix=".posture-review-data-", dir=MODEL_PATH.parent))
        try:
            for crop_path, label, source_name in labeled_rows:
                split = "val" if source_name in validation_sources else "train"
                destination_directory = prepared_root / split / label
                destination_directory.mkdir(parents=True, exist_ok=True)
                destination = destination_directory / f"{hashlib.sha256((source_name + str(crop_path)).encode('utf-8')).hexdigest()[:24]}{crop_path.suffix.lower()}"
                shutil.copy2(crop_path, destination)

            for split in ("train", "val"):
                for label in classes:
                    if not any((prepared_root / split / label).glob("*")):
                        raise PostureTrainingError(
                            f"No labeled images for posture class '{label}' in generated {split} split."
                        )
            return prepared_root
        except Exception:
            shutil.rmtree(prepared_root, ignore_errors=True)
            raise

    def start(
        self,
        dataset_id: str,
        *,
        epochs: int = DEFAULT_POSTURE_EPOCHS,
    ) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "running":
                raise PostureTrainingError("A posture training job is already running.")
            if not 1 <= epochs <= MAX_POSTURE_EPOCHS:
                raise PostureTrainingError(
                    f"Posture training epochs must be between 1 and {MAX_POSTURE_EPOCHS}."
                )
            dataset_root = get_uploaded_dataset(dataset_id)
            try:
                classification_root = self._classification_root(dataset_root)
                classification_root, cleanup_root = self._prepare_classification_dataset(classification_root)
            except PostureTrainingError as classification_error:
                prepared_root = self._prepare_coco_posture_dataset(dataset_root)
                if prepared_root is None:
                    prepared_root = self._prepare_review_manifest(dataset_root)
                if prepared_root is None:
                    raise classification_error
                classification_root = prepared_root
                cleanup_root = prepared_root
            self._state = {
                "status": "running",
                "dataset_id": dataset_id,
                "dataset_name": dataset_root.name,
                "epoch": 0,
                "epochs": epochs,
                "message": "Starting YOLO image-classification training.",
                "error": None,
            }
            thread = threading.Thread(
                target=self._train,
                args=(classification_root, epochs, cleanup_root),
                name="pig-posture-training",
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError as exc:
                if cleanup_root is not None:
                    shutil.rmtree(cleanup_root, ignore_errors=True)
                self._state.update(
                    status="failed",
                    message="Could not start posture classifier training.",
                    error=str(exc),
                )
                raise PostureTrainingError("Could not start posture classifier training.") from exc
            return dict(self._state)

    def _update(self, **updates: Any) -> None:
        with self._lock:
            self._state.update(updates)

    def _train(
        self,
        dataset_root: Path,
        epochs: int,
        cleanup_root: Path | None = None,
    ) -> None:
        temporary_root: Path | None = None
        candidate_path: Path | None = None
        try:
            from ultralytics import YOLO

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_root = Path(tempfile.mkdtemp(prefix="posture-training-", dir=MODEL_PATH.parent))
            run_directory = temporary_root / "run"
            candidate_path = temporary_root / "pig_posture.pt"
            model = YOLO("yolo11n-cls.pt")

            def record_epoch(trainer: Any) -> None:
                epoch = int(trainer.epoch) + 1
                self._update(
                    epoch=epoch,
                    epochs=int(trainer.epochs),
                    message=f"Training posture classifier, epoch {epoch} of {trainer.epochs}.",
                )

            model.add_callback("on_train_epoch_end", record_epoch)
            self._update(message="Training posture classifier. First run may download YOLO weights.")
            model.train(
                data=str(dataset_root),
                epochs=epochs,
                imgsz=IMAGE_SIZE,
                project=str(temporary_root),
                name="run",
                exist_ok=True,
                verbose=False,
                plots=False,
                workers=0,
                amp=False,
            )
            best_weights = run_directory / "weights" / "best.pt"
            if not best_weights.is_file():
                raise PostureTrainingError(
                    "Posture training completed without producing best.pt weights."
                )

            shutil.copy2(best_weights, candidate_path)
            trained_model = YOLO(str(candidate_path))
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate_path, MODEL_PATH)
            posture_detection_service.activate_model(trained_model)
            self._update(
                status="completed",
                epoch=epochs,
                message="Posture training complete. The new classifier is active.",
                error=None,
            )
        except Exception as exc:
            logger.exception("Posture classifier training failed.")
            self._update(
                status="failed",
                message="Posture training failed. Check the backend logs for details.",
                error=str(exc) or exc.__class__.__name__,
            )
        finally:
            if temporary_root is not None:
                shutil.rmtree(temporary_root, ignore_errors=True)
            if cleanup_root is not None:
                shutil.rmtree(cleanup_root, ignore_errors=True)
posture_training_service = PostureTrainingService()
