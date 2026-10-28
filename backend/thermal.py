import math
import threading
import time
from datetime import datetime, timezone
from typing import Any


THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24
THERMAL_PIXEL_COUNT = THERMAL_WIDTH * THERMAL_HEIGHT
THERMAL_FRAME_MAX_AGE_SECONDS = 10


class ThermalFrameService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pixels: list[float] | None = None
        self._ambient_c: float | None = None
        self._updated_at: float | None = None

    def ingest(self, pixels: list[float], ambient_c: float) -> None:
        if len(pixels) != THERMAL_PIXEL_COUNT:
            raise ValueError(f"Thermal frames must contain exactly {THERMAL_PIXEL_COUNT} pixels.")
        if not math.isfinite(ambient_c) or not -40 <= ambient_c <= 125:
            raise ValueError("Thermal ambient temperature must be between -40 and 125 C.")
        if any(not math.isfinite(pixel) or not -40 <= pixel <= 300 for pixel in pixels):
            raise ValueError("Thermal pixels must be finite temperatures between -40 and 300 C.")

        with self._lock:
            self._pixels = list(pixels)
            self._ambient_c = ambient_c
            self._updated_at = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            updated_at = self._updated_at
            ambient_c = self._ambient_c
            pixels = None if self._pixels is None else list(self._pixels)
        age_seconds = None if updated_at is None else max(0, time.time() - updated_at)
        online = age_seconds is not None and age_seconds <= THERMAL_FRAME_MAX_AGE_SECONDS
        return {
            "available": online,
            "sensor": "Waveshare MLX90640-D110",
            "width": THERMAL_WIDTH,
            "height": THERMAL_HEIGHT,
            "pixels": pixels if online else None,
            "ambient_c": None if ambient_c is None else round(ambient_c, 1),
            "updated_at": (
                None
                if updated_at is None
                else datetime.fromtimestamp(updated_at, timezone.utc).isoformat()
            ),
            "age_seconds": None if age_seconds is None else round(age_seconds, 1),
            "message": None if online else "Waiting for a recent MLX90640 frame from the ESP32-S3.",
        }

    def annotate_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._lock:
            pixels = self._pixels
            updated_at = self._updated_at
        if (
            pixels is None
            or updated_at is None
            or time.time() - updated_at > THERMAL_FRAME_MAX_AGE_SECONDS
        ):
            return detections

        for detection in detections:
            temperature = self._temperature_for_box(pixels, detection["box"])
            if temperature is not None:
                detection["surface_temperature_c"] = temperature
        return detections

    @staticmethod
    def _temperature_for_box(pixels: list[float], box: dict[str, float]) -> float | None:
        x = min(max(box["x"], 0), 100)
        y = min(max(box["y"], 0), 100)
        right = min(max(x + box["w"], 0), 100)
        bottom = min(max(y + box["h"], 0), 100)
        if right <= x or bottom <= y:
            return None

        left_px = max(0, math.floor(x * THERMAL_WIDTH / 100))
        top_px = max(0, math.floor(y * THERMAL_HEIGHT / 100))
        right_px = min(THERMAL_WIDTH, math.ceil(right * THERMAL_WIDTH / 100))
        bottom_px = min(THERMAL_HEIGHT, math.ceil(bottom * THERMAL_HEIGHT / 100))
        center_left = left_px + int((right_px - left_px) * 0.2)
        center_right = max(center_left + 1, right_px - int((right_px - left_px) * 0.2))
        center_top = top_px + int((bottom_px - top_px) * 0.15)
        center_bottom = max(center_top + 1, bottom_px - int((bottom_px - top_px) * 0.15))
        region = [
            pixels[row * THERMAL_WIDTH + column]
            for row in range(center_top, min(center_bottom, THERMAL_HEIGHT))
            for column in range(center_left, min(center_right, THERMAL_WIDTH))
        ]
        if not region:
            return None

        region.sort()
        warmest_count = max(1, math.ceil(len(region) * 0.25))
        warmest_median = region[-warmest_count:][warmest_count // 2]
        return round(warmest_median, 1)


thermal_frame_service = ThermalFrameService()
