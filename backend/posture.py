import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pig_posture.pt"
CONFIGURED_MODEL_PATH = Path(os.getenv("PIG_POSTURE_MODEL", str(DEFAULT_MODEL_PATH))).expanduser()
MODEL_PATH = (
    CONFIGURED_MODEL_PATH
    if CONFIGURED_MODEL_PATH.is_absolute()
    else (Path(__file__).resolve().parents[1] / CONFIGURED_MODEL_PATH).resolve()
)


class PostureModelError(RuntimeError):
    pass


class PostureDetectionService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._model_path = model_path
        self._model: Any = None
        self._model_source = "configured"
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._model_path.is_file():
            raise PostureModelError(
                "Posture identification needs trained YOLO classification weights. "
                "Place pig_posture.pt in backend/models or set PIG_POSTURE_MODEL."
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self._model_path))
        except Exception as exc:
            logger.exception("Could not load pig posture classification weights.")
            raise PostureModelError(
                "Could not load the posture classifier. Check that PIG_POSTURE_MODEL "
                "points to compatible YOLO classification weights."
            ) from exc
        return self._model

    @staticmethod
    def _class_names(names: Any) -> list[str]:
        if isinstance(names, dict):
            return [str(name) for _, name in sorted(names.items())]
        return [str(name) for name in names]

    def status(self) -> dict[str, Any]:
        try:
            with self._lock:
                model = self._load_model()
                names = self._class_names(model.names)
            return {
                "available": True,
                "classes": names,
                "message": None,
                "model_source": self._model_source,
            }
        except PostureModelError as exc:
            return {
                "available": False,
                "classes": [],
                "message": str(exc),
                "model_source": self._model_source,
            }

    def activate_model(self, model: Any) -> None:
        with self._lock:
            self._model = model
            self._model_source = "trained"

    def classify_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        height, width = frame.shape[:2]
        annotated = [dict(detection) for detection in detections]
        try:
            with self._lock:
                model = self._load_model()
                names = model.names
                for detection in annotated:
                    if detection.get("class_name") == "Unknown":
                        continue
                    box = detection["box"]
                    x1 = max(0, min(width, round(float(box["x"]) * width / 100)))
                    y1 = max(0, min(height, round(float(box["y"]) * height / 100)))
                    x2 = max(x1, min(width, round((float(box["x"]) + float(box["w"])) * width / 100)))
                    y2 = max(y1, min(height, round((float(box["y"]) + float(box["h"])) * height / 100)))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    results = model.predict(source=crop, verbose=False)
                    probabilities = results[0].probs if results else None
                    if probabilities is None:
                        raise PostureModelError(
                            "The configured posture model did not return classification probabilities. "
                            "Use YOLO image-classification weights."
                        )
                    class_id = int(probabilities.top1)
                    confidence = float(probabilities.top1conf.item())
                    if isinstance(names, dict):
                        label = names.get(class_id, str(class_id))
                    else:
                        label = names[class_id]
                    detection["posture"] = {
                        "label": str(label),
                        "confidence": round(confidence, 4),
                    }
            return {
                "detections": annotated,
                "posture_status": {
                    "available": True,
                    "classes": self._class_names(names),
                    "message": None,
                },
            }
        except PostureModelError as exc:
            return {
                "detections": annotated,
                "posture_status": {"available": False, "classes": [], "message": str(exc)},
            }
        except Exception as exc:
            logger.exception("Pig posture classification failed.")
            return {
                "detections": annotated,
                "posture_status": {
                    "available": False,
                    "classes": [],
                    "message": "Posture classification failed. Check the backend logs and model weights.",
                },
            }


posture_detection_service = PostureDetectionService()
