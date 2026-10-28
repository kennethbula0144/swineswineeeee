"""Create an unlabeled posture-review bundle from pig identity COCO annotations."""

import argparse
import csv
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
IDENTITY_CLASSES = {"pig 1", "pig 2", "pig 3", "pig 4"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class PostureReviewError(ValueError):
    pass


def _safe_source_path(root: Path, parent: Path, filename: str) -> Path:
    relative = PurePosixPath(filename)
    if relative.is_absolute() or "\\" in filename or ".." in relative.parts:
        raise PostureReviewError(f"Unsafe COCO image path: {filename}")
    path = (parent.joinpath(*relative.parts)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PostureReviewError(f"COCO image path escapes the dataset: {filename}") from exc
    return path


def create_review_bundle(source: Path, output: Path) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise PostureReviewError(f"Dataset source is not a directory: {source}")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise PostureReviewError("Output ZIP must be outside the source dataset directory.")

    annotation_paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for path in source.rglob("_annotations.coco.json"):
        split = SPLIT_NAMES.get(path.parent.name.lower())
        if split:
            annotation_paths[split].append(path)
    selected_paths = [
        path
        for split in ("train", "val", "test")
        for path in annotation_paths[split]
    ]
    if not selected_paths:
        raise PostureReviewError("No COCO train/validation/test annotations were found.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="posture-review-") as temp_dir:
        bundle_root = Path(temp_dir) / "posture-review"
        crops_root = bundle_root / "to_label"
        crops_root.mkdir(parents=True)
        manifest_path = bundle_root / "manifest.csv"
        image_count = 0
        warnings: list[str] = []
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=(
                    "crop_file",
                    "posture_label",
                    "pig_identity_from_source",
                    "source_split",
                    "source_image",
                    "source_annotation_id",
                ),
            )
            writer.writeheader()

            for annotation_path in selected_paths:
                try:
                    data = json.loads(annotation_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PostureReviewError(f"Could not parse COCO annotations: {annotation_path}") from exc
                if not isinstance(data, dict):
                    raise PostureReviewError(f"COCO annotation file must contain an object: {annotation_path}")
                categories = data.get("categories")
                images = data.get("images")
                annotations = data.get("annotations")
                if not isinstance(categories, list) or not isinstance(images, list) or not isinstance(annotations, list):
                    raise PostureReviewError(f"Invalid COCO categories/images/annotations in {annotation_path}")
                category_names = {
                    category.get("id"): category.get("name")
                    for category in categories
                    if isinstance(category, dict)
                }
                pig_categories = {
                    class_id: name
                    for class_id, name in category_names.items()
                    if isinstance(name, str) and name.strip().casefold() in IDENTITY_CLASSES
                }
                if not pig_categories:
                    warnings.append(f"No Pig 1–Pig 4 categories in {annotation_path.relative_to(source)}")
                    continue
                image_records = {
                    image.get("id"): image
                    for image in images
                    if isinstance(image, dict) and isinstance(image.get("id"), int)
                }
                split = SPLIT_NAMES[annotation_path.parent.name.lower()]
                for annotation in annotations:
                    if not isinstance(annotation, dict) or annotation.get("category_id") not in pig_categories:
                        continue
                    image_record = image_records.get(annotation.get("image_id"))
                    if image_record is None:
                        warnings.append(f"Skipped annotation with missing image record in {annotation_path.name}")
                        continue
                    filename = image_record.get("file_name")
                    if not isinstance(filename, str) or not filename:
                        warnings.append(f"Skipped annotation with missing image name in {annotation_path.name}")
                        continue
                    image_path = _safe_source_path(source, annotation_path.parent, filename)
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
                        warnings.append(f"Skipped missing image: {filename}")
                        continue
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        warnings.append(f"Skipped corrupt image: {filename}")
                        continue
                    height, width = image.shape[:2]
                    bbox = annotation.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        warnings.append(f"Skipped invalid box in {annotation_path.name}")
                        continue
                    try:
                        x, y, box_width, box_height = (float(value) for value in bbox)
                    except (ValueError, TypeError):
                        warnings.append(f"Skipped non-numeric box in {annotation_path.name}")
                        continue
                    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                        warnings.append(f"Skipped non-finite box in {annotation_path.name}")
                        continue
                    x1 = max(0, min(width, math.floor(x)))
                    y1 = max(0, min(height, math.floor(y)))
                    x2 = max(0, min(width, math.ceil(x + box_width)))
                    y2 = max(0, min(height, math.ceil(y + box_height)))
                    if x2 <= x1 or y2 <= y1:
                        warnings.append(f"Skipped empty box in {annotation_path.name}")
                        continue
                    crop = image[y1:y2, x1:x2]
                    crop_name = f"{split}_{image_count + 1:05d}.jpg"
                    crop_relative = Path("to_label") / crop_name
                    crop_path = bundle_root / crop_relative
                    if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        warnings.append(f"Could not encode crop from: {filename}")
                        continue
                    writer.writerow(
                        {
                            "crop_file": crop_relative.as_posix(),
                            "posture_label": "",
                            "pig_identity_from_source": pig_categories[annotation["category_id"]],
                            "source_split": split,
                            "source_image": filename,
                            "source_annotation_id": annotation.get("id", ""),
                        }
                    )
                    image_count += 1

        if not image_count:
            raise PostureReviewError("No valid pig identity crops were extracted from the COCO annotations.")
        (bundle_root / "README.txt").write_text(
            "POSTURE REVIEW CROPS — NOT LABELED FOR TRAINING YET\n\n"
            "These crops come from human-marked Pig 1–Pig 4 bounding boxes in the source COCO dataset. "
            "No posture labels have been inferred.\n\n"
            "Review each image in to_label/ and record its true posture in manifest.csv. "
            "Then split the labeled images into train/<posture-name>/ and val/<posture-name>/ "
            "folders (for example standing/, sternal_lying/, lateral_lying/). Keep images from the "
            "same source frame together in one split to reduce train/validation leakage. Exclude "
            "ambiguous, occluded, or unsuitable crops instead of guessing. The finished folder tree "
            "can be zipped and uploaded, then use Train posture in the dashboard.\n",
            encoding="utf-8",
        )

        extracted_size = sum(path.stat().st_size for path in bundle_root.rglob("*") if path.is_file())
        if extracted_size > MAX_EXTRACTED_BYTES:
            raise PostureReviewError("Review bundle exceeds the 4 GB uncompressed upload limit.")

        descriptor, temporary_zip_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        os.close(descriptor)
        temporary_zip = Path(temporary_zip_name)
        try:
            with zipfile.ZipFile(
                temporary_zip,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for path in sorted(bundle_root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(bundle_root).as_posix())
            compressed_size = temporary_zip.stat().st_size
            if compressed_size > MAX_ARCHIVE_BYTES:
                raise PostureReviewError("Review bundle exceeds the 512 MB compressed upload limit.")
            os.replace(temporary_zip, output)
        finally:
            temporary_zip.unlink(missing_ok=True)

    return {
        "output": output,
        "crops": image_count,
        "uncompressed_bytes": extracted_size,
        "compressed_bytes": output.stat().st_size,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted COCO dataset directory")
    parser.add_argument("-o", "--output", type=Path, help="Output ZIP path")
    args = parser.parse_args()
    output = args.output or args.source.expanduser().resolve().parent / "posture-review-crops.zip"
    try:
        result = create_review_bundle(args.source, output)
    except (PostureReviewError, OSError) as exc:
        parser.error(str(exc))
    print(f"Created posture review bundle: {result['output']}")
    print(f"Pig crops: {result['crops']}")
    print(f"Compressed: {result['compressed_bytes'] / (1024 * 1024):.1f} MB")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
