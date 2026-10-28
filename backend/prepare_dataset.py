"""Validate, normalize, and package detector or posture datasets for Smart Swine."""

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


class DatasetPreparationError(ValueError):
    pass


class DatasetPreparer:
    def __init__(self, source: Path, output: Path) -> None:
        self.source = source.expanduser().resolve(strict=True)
        self.output = output.expanduser().resolve()
        self.warnings: list[str] = []
        self.file_count = 0
        self.uncompressed_bytes = 0

        if not self.source.is_dir():
            raise DatasetPreparationError(f"Source must be an extracted dataset directory: {self.source}")
        try:
            self.output.relative_to(self.source)
        except ValueError:
            pass
        else:
            raise DatasetPreparationError("Output ZIP must be outside the source dataset directory.")

    def prepare(self, dataset_format: str) -> dict[str, Any]:
        mode = self._detect_format() if dataset_format == "auto" else dataset_format
        if mode not in {"coco", "yolo", "posture"}:
            raise DatasetPreparationError(f"Unsupported dataset format: {mode}")

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="smartswine-dataset-") as temporary_directory:
            prepared_root = Path(temporary_directory) / "dataset"
            prepared_root.mkdir()
            if mode == "coco":
                self._prepare_coco(prepared_root)
            elif mode == "yolo":
                self._prepare_yolo(prepared_root)
            else:
                self._prepare_posture(prepared_root)

            self._check_extracted_size(prepared_root)
            descriptor, temporary_zip_name = tempfile.mkstemp(
                prefix=f".{self.output.name}.",
                suffix=".tmp",
                dir=self.output.parent,
            )
            os.close(descriptor)
            temporary_zip = Path(temporary_zip_name)
            try:
                self._write_zip(prepared_root, temporary_zip)
                compressed_bytes = temporary_zip.stat().st_size
                if compressed_bytes > MAX_ARCHIVE_BYTES:
                    raise DatasetPreparationError(
                        f"Compressed ZIP is {self._format_size(compressed_bytes)}; "
                        f"the upload limit is {self._format_size(MAX_ARCHIVE_BYTES)}."
                    )
                os.replace(temporary_zip, self.output)
            finally:
                temporary_zip.unlink(missing_ok=True)

        return {
            "format": mode,
            "output": self.output,
            "files": self.file_count,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.output.stat().st_size,
            "warnings": self.warnings,
        }

    def _detect_format(self) -> str:
        if self._find_coco_splits():
            return "coco"
        if self._find_yolo_yaml():
            return "yolo"
        if self._find_posture_roots():
            return "posture"
        raise DatasetPreparationError(
            "Could not detect dataset format. Supply COCO annotations, one YOLO data.yaml, "
            "or posture train/<class>/ and val/<class>/ image folders."
        )

    def _find_coco_splits(self) -> dict[str, Path]:
        annotations: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        for annotation in self.source.rglob("_annotations.coco.json"):
            split = SPLIT_NAMES.get(annotation.parent.name.lower())
            if split:
                annotations[split].append(annotation)
        if not annotations["train"] and not annotations["val"]:
            return {}
        if len(annotations["train"]) != 1 or len(annotations["val"]) != 1:
            raise DatasetPreparationError(
                "COCO datasets require exactly one train and one val/valid "
                "_annotations.coco.json file."
            )
        for split, files in annotations.items():
            if len(files) > 1:
                raise DatasetPreparationError(f"COCO dataset has multiple {split} annotation files.")
        return {split: files[0] for split, files in annotations.items() if files}

    def _find_yolo_yaml(self) -> Path | None:
        configs = sorted(
            (path for path in self.source.rglob("*") if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}),
            key=lambda path: path.as_posix().lower(),
        )
        if len(configs) > 1:
            raise DatasetPreparationError("YOLO dataset must contain exactly one YAML configuration file.")
        return configs[0] if configs else None

    def _find_posture_roots(self) -> list[Path]:
        return [
            directory
            for directory in (self.source, *self.source.iterdir())
            if directory.is_dir()
            and (directory / "train").is_dir()
            and ((directory / "val").is_dir() or (directory / "valid").is_dir() or (directory / "validation").is_dir())
        ]

    def _resolve_inside_source(self, path: Path) -> Path:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.source)
        except ValueError as exc:
            raise DatasetPreparationError(f"Dataset path escapes the source directory: {path}") from exc
        return resolved

    def _candidate_inside_source(self, path: Path) -> Path | None:
        try:
            return self._resolve_inside_source(path)
        except DatasetPreparationError:
            self.warnings.append(f"Ignored dataset path outside the source directory: {path}")
            return None

    @staticmethod
    def _decode_image(path: Path) -> tuple[int, int] | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if not data:
            return None
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height

    def _copy_image(self, source: Path, destination: Path) -> bool:
        source = self._resolve_inside_source(source)
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            return False
        if self._decode_image(source) is None:
            self.warnings.append(f"Skipped missing or corrupt image: {source.relative_to(self.source)}")
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True

    def _prepare_posture(self, output: Path) -> None:
        roots = self._find_posture_roots()
        if len(roots) != 1:
            raise DatasetPreparationError("Posture datasets need one train and one val/valid/validation folder pair.")
        root = roots[0]
        validation_name = next(name for name in ("val", "valid", "validation") if (root / name).is_dir())
        train_classes = {path.name for path in (root / "train").iterdir() if path.is_dir()}
        val_classes = {path.name for path in (root / validation_name).iterdir() if path.is_dir()}
        if len(train_classes) < 2 or train_classes != val_classes:
            raise DatasetPreparationError("Train and validation splits must have at least two matching posture class folders.")

        copied_per_split = {
            (split, class_name): 0
            for split in ("train", "val")
            for class_name in train_classes
        }
        for source_split, output_split in (("train", "train"), (validation_name, "val")):
            for class_name in sorted(train_classes):
                source_class = root / source_split / class_name
                for image in source_class.rglob("*"):
                    if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    relative = image.relative_to(source_class)
                    target = output / output_split / class_name / relative
                    if self._copy_image(image, target):
                        copied_per_split[(output_split, class_name)] += 1

        for (split, class_name), count in copied_per_split.items():
            if count == 0:
                raise DatasetPreparationError(
                    f"No valid images remain in {split}/{class_name}."
                )

    def _prepare_yolo(self, output: Path) -> None:
        config_path = self._find_yolo_yaml()
        if config_path is None:
            raise DatasetPreparationError("No YOLO data.yaml/data.yml configuration was found.")
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise DatasetPreparationError(f"Could not parse YOLO YAML: {config_path}") from exc
        if not isinstance(config, dict):
            raise DatasetPreparationError("YOLO YAML must contain a mapping.")

        names_value = config.get("names")
        if isinstance(names_value, dict):
            try:
                names = [names_value[index] for index in range(len(names_value))]
            except KeyError as exc:
                raise DatasetPreparationError("YOLO class-name IDs must start at zero and be sequential.") from exc
        elif isinstance(names_value, list):
            names = names_value
        else:
            raise DatasetPreparationError("YOLO YAML must define class names as a list or indexed mapping.")
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise DatasetPreparationError("YOLO class names must be non-empty strings.")

        configured_base = config.get("path", ".")
        if not isinstance(configured_base, str):
            raise DatasetPreparationError("YOLO YAML 'path' must be a local folder path.")
        base_candidates = [
            candidate
            for candidate in (
                self._candidate_inside_source(config_path.parent / configured_base),
                self._candidate_inside_source(self.source / configured_base),
            )
            if candidate is not None
        ]
        base = next((candidate for candidate in base_candidates if candidate.is_dir()), None)
        if base is None:
            base = config_path.parent.resolve()

        output_counts: dict[str, int] = {}
        normalized: dict[str, Any] = {"path": ".", "names": [name.strip() for name in names], "nc": len(names)}
        for configured_split, output_split in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test")):
            if configured_split not in config:
                continue
            if output_split in output_counts:
                raise DatasetPreparationError("YOLO YAML defines duplicate val/valid split entries.")
            split_value = config[configured_split]
            if not isinstance(split_value, str):
                raise DatasetPreparationError(f"YOLO {configured_split} must reference one image directory.")

            path_candidates = [
                candidate
                for candidate in (
                    self._candidate_inside_source(base / split_value),
                    self._candidate_inside_source(config_path.parent / split_value),
                    self._candidate_inside_source(self.source / split_value),
                )
                if candidate is not None
            ]
            image_root = next((candidate for candidate in path_candidates if candidate.is_dir()), None)
            if image_root is None:
                self.warnings.append(f"Skipped missing {configured_split} image path: {split_value}")
                continue

            parts = list(image_root.parts)
            if "images" not in parts:
                raise DatasetPreparationError(
                    f"YOLO image path must contain an 'images' folder to locate matching labels: {image_root}"
                )
            image_index = len(parts) - 1 - parts[::-1].index("images")
            label_root = Path(*parts[:image_index], "labels", *parts[image_index + 1 :])
            label_root = self._resolve_inside_source(label_root)
            if not label_root.is_dir():
                raise DatasetPreparationError(f"Matching YOLO labels folder was not found: {label_root}")

            count = 0
            for image in image_root.rglob("*"):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image = self._resolve_inside_source(image)
                image_relative = image.relative_to(image_root)
                label_root = self._resolve_inside_source(label_root)
                label = (label_root / image_relative).with_suffix(".txt")
                label = self._resolve_inside_source(label)
                if not label.is_file():
                    self.warnings.append(
                        f"Skipped image without a matching YOLO label: {image.relative_to(self.source)}"
                    )
                    continue
                decoded = self._decode_image(image)
                if decoded is None:
                    self.warnings.append(f"Skipped corrupt image: {image.relative_to(self.source)}")
                    continue
                cleaned_labels: list[str] = []
                try:
                    label_lines = label.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError as exc:
                    raise DatasetPreparationError(f"Could not read YOLO label file: {label}") from exc
                for line_number, line in enumerate(label_lines, 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    try:
                        values = [float(field) for field in fields]
                        class_id = int(values[0])
                        valid = (
                            len(values) == 5
                            and values[0] == class_id
                            and 0 <= class_id < len(names)
                            and all(math.isfinite(value) for value in values[1:])
                            and all(0 <= value <= 1 for value in values[1:])
                            and values[3] > 0
                            and values[4] > 0
                        )
                    except (ValueError, IndexError):
                        valid = False
                    if valid:
                        cleaned_labels.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values[1:]))
                    else:
                        self.warnings.append(
                            f"Removed invalid label {label.relative_to(self.source)}:{line_number}"
                        )
                target_image = output / "images" / output_split / image_relative
                target_label = output / "labels" / output_split / image_relative.with_suffix(".txt")
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target_image)
                target_label.write_text("\n".join(cleaned_labels), encoding="utf-8")
                count += 1
            output_counts[output_split] = count
            normalized[output_split] = f"images/{output_split}"

        if "train" not in output_counts or "val" not in output_counts:
            raise DatasetPreparationError("YOLO dataset needs existing train and val/valid image folders.")
        if not output_counts["train"] or not output_counts["val"]:
            raise DatasetPreparationError("No valid labeled images remain in YOLO train or validation split.")
        if "test" in output_counts:
            normalized["test"] = "images/test"
        (output / "data.yaml").write_text(
            yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _prepare_coco(self, output: Path) -> None:
        annotations = self._find_coco_splits()
        if not annotations:
            raise DatasetPreparationError("No COCO train and validation annotation files were found.")
        split_images: dict[str, dict[int, dict[str, Any]]] = {}
        category_names: list[str] | None = None

        for split, annotation_path in annotations.items():
            try:
                data = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetPreparationError(f"Could not parse COCO annotations: {annotation_path}") from exc
            if not isinstance(data, dict):
                raise DatasetPreparationError(f"COCO annotation file must contain an object: {annotation_path}")
            categories, images, records = data.get("categories"), data.get("images"), data.get("annotations")
            if not isinstance(categories, list) or not categories or not isinstance(images, list) or not isinstance(records, list):
                raise DatasetPreparationError(f"COCO categories, images, or annotations are invalid: {annotation_path}")
            categories = sorted(categories, key=lambda item: item.get("id", -1) if isinstance(item, dict) else -1)
            current_names: list[str] = []
            category_ids: dict[int, int] = {}
            for index, category in enumerate(categories):
                if not isinstance(category, dict) or not isinstance(category.get("id"), int):
                    raise DatasetPreparationError("COCO category IDs must be integers.")
                name = category.get("name")
                if not isinstance(name, str) or not name.strip() or category["id"] in category_ids:
                    raise DatasetPreparationError("COCO categories need unique IDs and non-empty names.")
                category_ids[category["id"]] = index
                current_names.append(name.strip())
            if category_names is None:
                category_names = current_names
            elif current_names != category_names:
                raise DatasetPreparationError("COCO train, validation, and test class names do not match.")

            image_records: dict[int, dict[str, Any]] = {}
            for image in images:
                if not isinstance(image, dict) or not isinstance(image.get("id"), int):
                    self.warnings.append(f"Skipped invalid COCO image record in {annotation_path.name}")
                    continue
                filename = image.get("file_name")
                if not isinstance(filename, str) or not filename:
                    self.warnings.append(f"Skipped image record with no file name in {annotation_path.name}")
                    continue
                relative_path = PurePosixPath(filename)
                if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in filename:
                    self.warnings.append(f"Skipped unsafe COCO image path: {filename}")
                    continue
                source_image = self._resolve_inside_source(annotation_path.parent.joinpath(*relative_path.parts))
                dimensions = self._decode_image(source_image)
                if dimensions is None:
                    self.warnings.append(f"Skipped missing or corrupt COCO image: {filename}")
                    continue
                image_records[image["id"]] = {
                    "source": source_image,
                    "filename": Path(*relative_path.parts),
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "labels": [],
                }

            for record in records:
                if not isinstance(record, dict):
                    self.warnings.append(f"Skipped malformed COCO annotation in {annotation_path.name}")
                    continue
                image_record = image_records.get(record.get("image_id"))
                bbox = record.get("bbox")
                class_id = category_ids.get(record.get("category_id"))
                if image_record is None or class_id is None:
                    self.warnings.append(f"Skipped COCO annotation with missing image or unknown category in {annotation_path.name}")
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    self.warnings.append(f"Skipped malformed COCO box in {annotation_path.name}")
                    continue
                try:
                    x, y, box_width, box_height = (float(value) for value in bbox)
                except (ValueError, TypeError):
                    self.warnings.append(f"Skipped non-numeric COCO box in {annotation_path.name}")
                    continue
                if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                    self.warnings.append(f"Skipped non-finite COCO box in {annotation_path.name}")
                    continue
                width, height = image_record["width"], image_record["height"]
                x1, y1 = max(0.0, min(float(width), x)), max(0.0, min(float(height), y))
                x2 = max(0.0, min(float(width), x + box_width))
                y2 = max(0.0, min(float(height), y + box_height))
                if x2 <= x1 or y2 <= y1:
                    self.warnings.append(f"Skipped empty COCO box in {annotation_path.name}")
                    continue
                image_record["labels"].append(
                    f"{class_id} {(x1 + x2) / (2 * width):.6f} {(y1 + y2) / (2 * height):.6f} "
                    f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
                )

            split_images[split] = image_records

        for split, images in split_images.items():
            valid_images = 0
            for image_record in images.values():
                relative = image_record["filename"]
                image_target = output / "images" / split / relative
                label_target = output / "labels" / split / relative.with_suffix(".txt")
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_record["source"], image_target)
                label_target.write_text("\n".join(image_record["labels"]), encoding="utf-8")
                valid_images += 1
            if split in {"train", "val"} and valid_images == 0:
                raise DatasetPreparationError(f"No valid images remain in the COCO {split} split.")

        config: dict[str, Any] = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "names": category_names,
            "nc": len(category_names or []),
        }
        if "test" in split_images:
            config["test"] = "images/test"
        (output / "data.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _check_extracted_size(self, root: Path) -> None:
        total = 0
        file_count = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise DatasetPreparationError(f"Prepared dataset contains a symbolic link: {path}")
            if path.is_file():
                file_count += 1
                total += path.stat().st_size
                if total > MAX_EXTRACTED_BYTES:
                    raise DatasetPreparationError(
                        f"Uncompressed dataset exceeds the {self._format_size(MAX_EXTRACTED_BYTES)} limit."
                    )
        self.file_count = file_count
        self.uncompressed_bytes = total

    @staticmethod
    def _write_zip(source: Path, destination: Path) -> None:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    relative_path = path.relative_to(source)
                    archive_name = relative_path.as_posix()
                    if (
                        "\\" in archive_name
                        or any(part in {"", ".", ".."} for part in PurePosixPath(archive_name).parts)
                        or any(
                            any(character in part for character in '<>:"|?*')
                            or part.endswith((" ", "."))
                            or part.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL"}
                            or re.fullmatch(r"(COM|LPT)[1-9]", part.split(".", 1)[0].upper())
                            for part in PurePosixPath(archive_name).parts
                        )
                    ):
                        raise DatasetPreparationError(
                            f"File path cannot be safely uploaded on Windows: {archive_name}"
                        )
                    archive.write(path, archive_name)

    @staticmethod
    def _format_size(size: int) -> str:
        return f"{size / (1024 * 1024):.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted dataset directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output ZIP path (default: dataset.zip beside the source directory)",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "detector", "posture"),
        default="auto",
        help="Dataset task; auto detects input. detector accepts COCO or YOLO.",
    )
    args = parser.parse_args()
    output = args.output or args.source.expanduser().resolve().parent / "dataset.zip"
    mode = {"detector": "yolo"}.get(args.format, args.format)
    try:
        result = DatasetPreparer(args.source, output).prepare(mode)
    except (DatasetPreparationError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))

    print(f"Prepared {result['format']} dataset: {result['output']}")
    print(f"Files: {result['files']}")
    print(f"Uncompressed: {DatasetPreparer._format_size(result['uncompressed_bytes'])}")
    print(f"Compressed: {DatasetPreparer._format_size(result['compressed_bytes'])}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
