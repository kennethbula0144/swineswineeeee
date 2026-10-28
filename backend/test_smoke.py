import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pig_monitoring import PigDetection, PigDetectionResult, detect_pigs


class PigMonitoringSmokeTests(unittest.TestCase):
    def test_detect_pigs_returns_attributes_used_by_public_example(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "pen.jpg"
            image.write_bytes(b"sample encoded frame")
            detector_result = {
                "width": 640,
                "height": 480,
                "inference_ms": 24.5,
                "detections": [
                    {
                        "class_name": "pig",
                        "confidence": 0.91,
                        "box": {"x": 12.5, "y": 10.0, "w": 25.0, "h": 30.0},
                    }
                ],
            }
            with patch("pig_monitoring.pig_detection_service.detect", return_value=detector_result):
                result = detect_pigs(image)

        self.assertIsInstance(result, PigDetectionResult)
        self.assertEqual(result.image_width, 640)
        self.assertEqual(result.image_height, 480)
        self.assertEqual(result.inference_ms, 24.5)
        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertIsInstance(detection, PigDetection)
        self.assertEqual(
            (
                detection.class_name,
                detection.confidence,
                detection.x,
                detection.y,
                detection.width,
                detection.height,
            ),
            ("pig", 0.91, 80.0, 48.0, 160.0, 144.0),
        )

    def test_detect_pigs_reports_missing_image_path(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "missing.jpg"
            with self.assertRaisesRegex(FileNotFoundError, "Pig detection image was not found"):
                detect_pigs(image)


if __name__ == "__main__":
    unittest.main()
