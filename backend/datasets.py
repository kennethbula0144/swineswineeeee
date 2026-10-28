import json
import logging
import os
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).resolve().parents[1] / "aitrain" / "datasets"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 100_000
CHUNK_BYTES = 1024 * 1024
UPLOAD_MARKER = ".smartswine-upload.json"
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"}


def _is_safe_member_path(filename: str) -> bool:
    member_path = PurePosixPath(filename)
    if (
        not filename
        or "\\" in filename
        or member_path.is_absolute()
        or any(part in {"", ".", ".."} for part in member_path.parts)
    ):
        return False
    for part in member_path.parts:
        if any(character in part for character in '<>:"|?*') or part.endswith((" ", ".")):
            return False
        device_name = part.split(".", 1)[0].upper()
        if device_name in WINDOWS_RESERVED_NAMES or re.fullmatch(r"(COM|LPT)[1-9]", device_name):
            return False
    return True


class DatasetArchiveError(ValueError):
    pass


class DatasetSizeError(DatasetArchiveError):
    pass


class DatasetNotFoundError(FileNotFoundError):
    pass


def get_uploaded_dataset(dataset_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", dataset_id):
        raise DatasetNotFoundError("Uploaded dataset was not found.")
    root = DATASET_ROOT.resolve()
    directory = DATASET_ROOT / dataset_id
    try:
        resolved_directory = directory.resolve(strict=True)
        resolved_directory.relative_to(root)
    except (OSError, ValueError) as exc:
        raise DatasetNotFoundError("Uploaded dataset was not found.") from exc
    if (
        resolved_directory.parent != root
        or not resolved_directory.is_dir()
        or not (resolved_directory / UPLOAD_MARKER).is_file()
    ):
        raise DatasetNotFoundError("Uploaded dataset was not found.")
    try:
        metadata = json.loads(
            (resolved_directory / UPLOAD_MARKER).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetNotFoundError("Uploaded dataset was not found.") from exc
    if metadata.get("id") != dataset_id or metadata.get("folder") != dataset_id:
        raise DatasetNotFoundError("Uploaded dataset was not found.")
    return resolved_directory


def delete_uploaded_dataset(dataset_id: str) -> None:
    directory = get_uploaded_dataset(dataset_id)
    shutil.rmtree(directory)


def list_uploaded_datasets() -> list[dict[str, Any]]:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    uploads = []
    for directory in DATASET_ROOT.iterdir():
        marker_path = directory / UPLOAD_MARKER
        if not directory.is_dir() or not marker_path.is_file():
            continue
        try:
            metadata = json.loads(marker_path.read_text(encoding="utf-8"))
            uploads.append(metadata)
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable dataset upload metadata at %s.", marker_path)
    return sorted(uploads, key=lambda item: item["uploaded_at"], reverse=True)


def save_dataset_archive(archive_file: BinaryIO, original_filename: str) -> dict[str, Any]:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    archive_file.seek(0, os.SEEK_END)
    archive_size = archive_file.tell()
    archive_file.seek(0)
    if archive_size == 0:
        raise DatasetArchiveError("The selected ZIP file is empty.")
    if archive_size > MAX_ARCHIVE_BYTES:
        raise DatasetSizeError("ZIP files must be 512 MB or smaller.")

    if not original_filename.lower().endswith(".zip"):
        raise DatasetArchiveError("Upload a dataset archive with a .zip extension.")
    if not zipfile.is_zipfile(archive_file):
        raise DatasetArchiveError("The selected file is not a valid ZIP archive.")
    archive_file.seek(0)

    archive_name = Path(original_filename).name
    dataset_name = Path(archive_name).stem
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset_name).strip(".-_")[:60]
    safe_name = safe_name or "dataset"
    uploaded_at = datetime.now(timezone.utc).isoformat()
    upload_id = uuid.uuid4().hex[:8]
    folder_name = f"{safe_name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{upload_id}"
    staging_path = Path(tempfile.mkdtemp(prefix=".dataset-upload-", dir=DATASET_ROOT))
    destination = DATASET_ROOT / folder_name

    try:
        with zipfile.ZipFile(archive_file) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_FILES:
                raise DatasetSizeError("The archive contains too many files.")
            declared_size = sum(member.file_size for member in members)
            if declared_size > MAX_EXTRACTED_BYTES:
                raise DatasetSizeError("The extracted dataset must be 4 GB or smaller.")
            if not members or not any(not member.is_dir() for member in members):
                raise DatasetArchiveError("The ZIP archive does not contain any files.")

            extracted_size = 0
            extracted_files = 0
            for member in members:
                member_path = PurePosixPath(member.filename)
                if not _is_safe_member_path(member.filename):
                    raise DatasetArchiveError("The ZIP archive contains an unsafe file path.")
                if member.flag_bits & 0x1:
                    raise DatasetArchiveError("Password-protected ZIP files are not supported.")

                unix_mode = member.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                if file_type and file_type not in {stat.S_IFDIR, stat.S_IFREG}:
                    raise DatasetArchiveError("The ZIP archive contains an unsupported file type.")

                target = staging_path.joinpath(*member_path.parts)
                if member.is_dir():
                    if target.exists() and not target.is_dir():
                        raise DatasetArchiveError("The ZIP archive contains conflicting file paths.")
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                if target.exists():
                    raise DatasetArchiveError("The ZIP archive contains conflicting file paths.")
                for parent in target.parents:
                    if parent == staging_path:
                        break
                    if parent.exists() and not parent.is_dir():
                        raise DatasetArchiveError("The ZIP archive contains conflicting file paths.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    while chunk := source.read(CHUNK_BYTES):
                        extracted_size += len(chunk)
                        if extracted_size > MAX_EXTRACTED_BYTES:
                            raise DatasetSizeError("The extracted dataset must be 4 GB or smaller.")
                        output.write(chunk)
                extracted_files += 1

        metadata = {
            "id": folder_name,
            "name": archive_name,
            "folder": folder_name,
            "uploaded_at": uploaded_at,
            "file_count": extracted_files,
            "size_bytes": extracted_size,
        }
        (staging_path / UPLOAD_MARKER).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        staging_path.rename(destination)
        return metadata
    except (
        DatasetArchiveError,
        EOFError,
        NotImplementedError,
        RuntimeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as exc:
        shutil.rmtree(staging_path, ignore_errors=True)
        if isinstance(exc, DatasetArchiveError):
            raise
        raise DatasetArchiveError("Could not read the ZIP archive. Check that it is valid and try again.") from exc
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        logger.exception("Could not save uploaded dataset archive.")
        raise
