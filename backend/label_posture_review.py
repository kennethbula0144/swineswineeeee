"""Interactively label pig-crop review ZIPs without guessing posture labels."""

import argparse
import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_CLASSES = (
    "standing",
    "sitting",
    "sternal_lying",
    "lateral_lying",
)


class PostureLabelerError(ValueError):
    pass


def _safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _read_review_manifest(archive: zipfile.ZipFile) -> tuple[str, list[dict[str, str]]]:
    manifest_names = [
        name
        for name in archive.namelist()
        if PurePosixPath(name).name.casefold() == "manifest.csv"
    ]
    if len(manifest_names) != 1:
        raise PostureLabelerError("Review ZIP must contain exactly one manifest.csv.")
    manifest_name = manifest_names[0]
    try:
        reader = csv.DictReader(io.StringIO(archive.read(manifest_name).decode("utf-8-sig")))
        required = {"crop_file", "posture_label", "source_image"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PostureLabelerError(
                "manifest.csv needs crop_file, posture_label, and source_image columns."
            )
        rows = [dict(row) for row in reader]
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise PostureLabelerError("Could not read manifest.csv from the review ZIP.") from exc
    for line_number, row in enumerate(rows, 2):
        crop_file = row.get("crop_file", "")
        if not _safe_zip_name(crop_file):
            raise PostureLabelerError(f"Unsafe crop_file path in manifest.csv line {line_number}.")
        if crop_file not in archive.namelist():
            raise PostureLabelerError(f"Crop image is missing from ZIP: {crop_file}")
    return manifest_name, rows


def write_labeled_review_zip(
    source_zip: Path,
    output_zip: Path,
    manifest_name: str,
    rows: list[dict[str, str]],
) -> None:
    output_zip = output_zip.expanduser().resolve()
    source_zip = source_zip.expanduser().resolve(strict=True)
    if output_zip == source_zip:
        raise PostureLabelerError("Output ZIP must not overwrite the source review ZIP.")
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_zip.name}.",
        suffix=".tmp",
        dir=output_zip.parent,
    )
    os.close(descriptor)
    temporary_zip = Path(temporary_name)
    try:
        with zipfile.ZipFile(source_zip, "r") as source, zipfile.ZipFile(
            temporary_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as output:
            entries = source.infolist()
            if sum(entry.file_size for entry in entries) > MAX_EXTRACTED_BYTES:
                raise PostureLabelerError("Review ZIP exceeds the 4 GB uncompressed limit.")
            updated_manifest = io.StringIO(newline="")
            writer = csv.DictWriter(
                updated_manifest,
                fieldnames=rows[0].keys() if rows else (),
                lineterminator="\n",
            )
            if rows:
                writer.writeheader()
                writer.writerows(rows)
            for entry in entries:
                if entry.is_dir():
                    continue
                if not _safe_zip_name(entry.filename):
                    raise PostureLabelerError(f"Unsafe path in review ZIP: {entry.filename}")
                if entry.filename == manifest_name:
                    data = updated_manifest.getvalue().encode("utf-8-sig")
                else:
                    data = source.read(entry)
                output.writestr(entry.filename, data)
        if temporary_zip.stat().st_size > MAX_ARCHIVE_BYTES:
            raise PostureLabelerError("Labeled review ZIP exceeds the 512 MB compressed upload limit.")
        os.replace(temporary_zip, output_zip)
    finally:
        temporary_zip.unlink(missing_ok=True)


class PostureLabeler:
    def __init__(
        self,
        source_zip: Path,
        output_zip: Path,
        classes: tuple[str, ...],
    ) -> None:
        self.source_zip = source_zip.expanduser().resolve(strict=True)
        self.output_zip = output_zip.expanduser().resolve()
        if self.source_zip.stat().st_size > MAX_ARCHIVE_BYTES:
            raise PostureLabelerError("Review ZIP exceeds the 512 MB compressed upload limit.")
        self.archive = zipfile.ZipFile(self.source_zip)
        if sum(entry.file_size for entry in self.archive.infolist()) > MAX_EXTRACTED_BYTES:
            self.archive.close()
            raise PostureLabelerError("Review ZIP exceeds the 4 GB uncompressed upload limit.")
        self.manifest_name, self.rows = _read_review_manifest(self.archive)
        self.classes = classes
        self.sidecar = self.source_zip.with_suffix(".labels.csv")
        self.index = self._first_unlabeled_index()
        self.root: Any = None

        if self.sidecar.exists():
            try:
                with self.sidecar.open(newline="", encoding="utf-8-sig") as progress_file:
                    saved_rows = list(csv.DictReader(progress_file))
                if len(saved_rows) == len(self.rows):
                    for row, saved in zip(self.rows, saved_rows):
                        label = (saved.get("posture_label") or "").strip()
                        if not label or label in self.classes:
                            row["posture_label"] = label
                    self.index = self._first_unlabeled_index()
            except (OSError, UnicodeDecodeError, csv.Error):
                raise PostureLabelerError(f"Could not read saved labeling progress: {self.sidecar}")

    def _first_unlabeled_index(self) -> int:
        return next(
            (index for index, row in enumerate(self.rows) if not row.get("posture_label", "").strip()),
            max(0, len(self.rows) - 1),
        )

    def _save_progress(self) -> None:
        with self.sidecar.open("w", newline="", encoding="utf-8-sig") as progress_file:
            writer = csv.DictWriter(progress_file, fieldnames=self.rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)

    def run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox

            from PIL import Image, ImageTk
        except ImportError as exc:
            self.archive.close()
            raise PostureLabelerError(
                "The visual labeler requires Tkinter and Pillow. Install Pillow with "
                "'python -m pip install pillow'."
            ) from exc

        self.root = tk.Tk()
        self.root.title("Smart Swine - Label pig posture crops")
        self.root.geometry("1100x820")
        self.root.minsize(720, 560)
        self.status = tk.StringVar()
        tk.Label(
            self.root,
            text="Choose the posture visible in this crop. Labels are your manual review, not an AI guess.",
            wraplength=1000,
            justify="left",
        ).pack(padx=16, pady=(14, 6), anchor="w")
        self.image_label = tk.Label(self.root, text="Loading crop…")
        self.image_label.pack(fill="both", expand=True, padx=16, pady=8)
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x", padx=16, pady=4)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=(4, 12))
        for key, label in enumerate(self.classes, 1):
            tk.Button(
                controls,
                text=f"{key}: {label}",
                command=lambda selected=label: self._label_current(selected, messagebox),
            ).pack(side="left", padx=3, pady=3)
        tk.Button(
            controls,
            text="S: Skip / leave blank",
            command=lambda: self._label_current("", messagebox),
        ).pack(side="left", padx=3, pady=3)
        tk.Button(
            controls,
            text="Save ZIP and close",
            command=lambda: self._finish(messagebox),
        ).pack(side="right", padx=3, pady=3)
        tk.Button(
            controls,
            text="Previous",
            command=self._previous,
        ).pack(side="right", padx=3, pady=3)
        self.root.bind("<Key>", lambda event: self._key_pressed(event, messagebox))
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._finish(messagebox))
        self._show_current(Image, ImageTk)
        self.root.mainloop()
        self.archive.close()

    def _key_pressed(self, event: Any, messagebox: Any) -> None:
        if event.char.isdigit() and 1 <= int(event.char) <= len(self.classes):
            self._label_current(self.classes[int(event.char) - 1], messagebox)
        elif event.char.lower() in {"s", " "}:
            self._label_current("", messagebox)
        elif event.keysym == "Left":
            self._previous()
        elif event.keysym == "Right":
            self._label_current("", messagebox)

    def _show_current(self, image_module: Any, image_tk_module: Any) -> None:
        if not self.rows:
            self.status.set("The review manifest contains no crops.")
            return
        row = self.rows[self.index]
        image_bytes = self.archive.read(row["crop_file"])
        image = image_module.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((1000, 620))
        self.photo = image_tk_module.PhotoImage(image)
        self.image_label.configure(image=self.photo, text="")
        label = row.get("posture_label", "").strip() or "Unlabeled"
        self.status.set(
            f"Crop {self.index + 1} of {len(self.rows)} · Current label: {label} · "
            f"Pig identity: {row.get('pig_identity_from_source', 'unknown')} · "
            f"Source: {row.get('source_image', 'unknown')}"
        )

    def _refresh(self) -> None:
        from PIL import Image, ImageTk

        self._show_current(Image, ImageTk)

    def _label_current(self, label: str, messagebox: Any) -> None:
        if not self.rows:
            return
        self.rows[self.index]["posture_label"] = label
        try:
            self._save_progress()
        except OSError as exc:
            messagebox.showerror("Could not save progress", str(exc))
            return
        if self.index < len(self.rows) - 1:
            self.index += 1
        self._refresh()

    def _previous(self) -> None:
        if self.rows and self.index > 0:
            self.index -= 1
            self._refresh()

    def _finish(self, messagebox: Any) -> None:
        try:
            self._save_progress()
            write_labeled_review_zip(
                self.source_zip,
                self.output_zip,
                self.manifest_name,
                self.rows,
            )
        except (OSError, PostureLabelerError, zipfile.BadZipFile) as exc:
            messagebox.showerror("Could not export labeled ZIP", str(exc))
            return
        labeled = sum(bool(row.get("posture_label", "").strip()) for row in self.rows)
        messagebox.showinfo(
            "Labeled review ZIP saved",
            f"Saved {labeled} labeled crops to:\n{self.output_zip}\n\n"
            "Upload this ZIP and choose Train posture. At least two different posture classes "
            "with crops from multiple source frames are required.",
        )
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_zip", type=Path, help="Generated posture-review-crops.zip")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Labeled ZIP path (default: <review-name>-labeled.zip)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Posture labels shown as buttons in the UI.",
    )
    args = parser.parse_args()
    output = args.output or args.review_zip.with_name(f"{args.review_zip.stem}-labeled.zip")
    try:
        PostureLabeler(args.review_zip, output, tuple(args.classes)).run()
    except (OSError, PostureLabelerError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
