from dataclasses import dataclass
from pathlib import Path

from detection import (
    MAX_FRAME_BYTES,
    DetectorUnavailableError,
    pig_detection_service,
)


@dataclass(frozen=True, slots=True)
class PigDetection:
    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class PigDetectionResult:
    detections: tuple[PigDetection, ...]
    image_width: int
    image_height: int
    inference_ms: float


def detect_pigs(image_path: str | Path) -> PigDetectionResult:
    """Detect pigs in an image and return pixel-coordinate bounding boxes."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pig detection image was not found: {path}")

    with path.open("rb") as image_file:
        frame = image_file.read(MAX_FRAME_BYTES + 1)
    result = pig_detection_service.detect(frame)
    width = result["width"]
    height = result["height"]
    detections = tuple(
        PigDetection(
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            x=round(detection["box"]["x"] * width / 100, 2),
            y=round(detection["box"]["y"] * height / 100, 2),
            width=round(detection["box"]["w"] * width / 100, 2),
            height=round(detection["box"]["h"] * height / 100, 2),
        )
        for detection in result["detections"]
    )
    return PigDetectionResult(
        detections=detections,
        image_width=width,
        image_height=height,
        inference_ms=result["inference_ms"],
    )


__all__ = [
    "DetectorUnavailableError",
    "PigDetection",
    "PigDetectionResult",
    "detect_pigs",
]
