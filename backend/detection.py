import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from posture import posture_detection_service

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_DETECTIONS = 4
INFERENCE_IMAGE_SIZE = 960
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pig_detector.pt"
CONFIGURED_MODEL_PATH = Path(os.getenv("PIG_DETECTOR_MODEL", str(DEFAULT_MODEL_PATH))).expanduser()
MODEL_PATH = (
    CONFIGURED_MODEL_PATH
    if CONFIGURED_MODEL_PATH.is_absolute()
    else (Path(__file__).resolve().parents[1] / CONFIGURED_MODEL_PATH).resolve()
)
CONFIDENCE_THRESHOLD = float(os.getenv("PIG_DETECTOR_CONFIDENCE", "0.35"))

if not math.isfinite(CONFIDENCE_THRESHOLD) or not 0 <= CONFIDENCE_THRESHOLD <= 1:
    raise ValueError("PIG_DETECTOR_CONFIDENCE must be between 0 and 1.")


class DetectorUnavailableError(RuntimeError):
    pass


class PigDetectionService:
    def __init__(self) -> None:
        self._model: Any = None
        self._model_source = "configured"
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not MODEL_PATH.is_file():
            raise DetectorUnavailableError(
                "Pig detector weights were not found. Place pig_detector.pt in backend/models "
                "or set PIG_DETECTOR_MODEL to the weights path."
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(MODEL_PATH))
        except Exception as exc:
            logger.exception("Could not load pig detector weights.")
            raise DetectorUnavailableError(
                "Could not load the pig detector. Check the YOLO backend dependencies and model weights."
            ) from exc
        return self._model

    @staticmethod
    def _pig_class_ids(names: Any) -> list[int]:
        class_names = names.items() if isinstance(names, dict) else enumerate(names)
        return [
            class_id
            for class_id, name in class_names
            if isinstance(name, str)
            and re.fullmatch(r"pig(?:[\s_-]*\d+)?", name.strip(), flags=re.IGNORECASE)
        ]

    def status(self) -> dict[str, Any]:
        try:
            with self._lock:
                model = self._load_model()
                names = model.names
                pig_class_ids = self._pig_class_ids(names)
                classes = [
                    names[class_id] if isinstance(names, dict) else names[class_id]
                    for class_id in pig_class_ids
                ]
                if not classes:
                    raise DetectorUnavailableError(
                        "The configured detector has no pig or pig-identity classes. "
                        "Train it with labels such as pig or Pig 1 through Pig 4."
                    )
                classes.append("Unknown")
            return {
                "available": True,
                "classes": classes,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "model_source": self._model_source,
            }
        except DetectorUnavailableError as exc:
            return {
                "available": False,
                "classes": [],
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "model_source": self._model_source,
                "message": str(exc),
            }

    def activate_model(self, model: Any) -> None:
        with self._lock:
            self._model = model
            self._model_source = "trained"

    def detect(self, frame_bytes: bytes) -> dict[str, Any]:
        if not frame_bytes:
            raise ValueError("The uploaded frame is empty.")
        if len(frame_bytes) > MAX_FRAME_BYTES:
            raise ValueError("Image is too large. Frames must be 8 MB or smaller.")

        encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the image. Upload a JPEG, PNG, or WebP frame.")

        height, width = frame.shape[:2]
        started_at = time.perf_counter()
        with self._lock:
            model = self._load_model()
            names = model.names
            pig_class_ids = self._pig_class_ids(names)
            if not pig_class_ids:
                raise DetectorUnavailableError(
                    "The configured detector has no pig or pig-identity classes. "
                    "Train it with labels such as pig or Pig 1 through Pig 4."
                )
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=INFERENCE_IMAGE_SIZE,
                max_det=MAX_DETECTIONS,
                classes=pig_class_ids,
                agnostic_nms=True,
                verbose=False,
            )

        detections = []
        if results:
            result = results[0]
            names = result.names
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                    x1 = min(max(x1, 0), width)
                    x2 = min(max(x2, 0), width)
                    y1 = min(max(y1, 0), height)
                    y2 = min(max(y2, 0), height)
                    if x2 <= x1 or y2 <= y1 or not math.isfinite(confidence):
                        continue
                    if class_id not in pig_class_ids:
                        continue
                    detected_class = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    detections.append(
                        {
                            "class_name": str(detected_class),
                            "confidence": round(confidence, 4),
                            "box": {
                                "x": round(x1 * 100 / width, 2),
                                "y": round(y1 * 100 / height, 2),
                                "w": round((x2 - x1) * 100 / width, 2),
                                "h": round((y2 - y1) * 100 / height, 2),
                            },
                        }
                    )

        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        detections = detections[:MAX_DETECTIONS]
        posture_result = posture_detection_service.classify_detections(frame, detections)
        return {
            "width": width,
            "height": height,
            "inference_ms": round((time.perf_counter() - started_at) * 1000, 2),
            **posture_result,
        }


pig_detection_service = PigDetectionService()
