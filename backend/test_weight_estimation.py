import unittest
from unittest.mock import patch

import httpx

from app import app
from weight_estimation import add_weight_estimates, estimate_weight_from_detection


class WeightEstimationTests(unittest.TestCase):
    def test_estimates_weight_automatically_from_detection_box(self):
        detection = {
            "class_name": "pig",
            "confidence": 0.9,
            "box": {"x": 0, "y": 0, "w": 25, "h": 50},
        }

        estimate = estimate_weight_from_detection(
            detection,
            image_width=640,
            image_height=480,
        )

        self.assertAlmostEqual(estimate["girth_proxy_inches"], 39.37, places=2)
        self.assertAlmostEqual(estimate["body_length_proxy_inches"], 45.46, places=2)
        self.assertAlmostEqual(estimate["weight_pounds"], 176.16, places=2)
        self.assertAlmostEqual(estimate["weight_kg"], 79.91, places=2)
        self.assertEqual(estimate["method"], "overhead_bbox_fixed_camera_proxy")

    def test_estimates_detection_without_camera_setup(self):
        result = {
            "width": 640,
            "height": 480,
            "detections": [{"box": {"x": 1, "y": 2, "w": 3, "h": 4}}],
        }

        estimated = add_weight_estimates(result)
        self.assertIn("weight_estimate", estimated["detections"][0])

    def test_unknown_detection_is_preserved_without_pig_weight_estimate(self):
        result = {
            "width": 640,
            "height": 480,
            "detections": [{"class_name": "Unknown", "box": {"x": 1, "y": 2, "w": 3, "h": 4}}],
        }

        estimated = add_weight_estimates(result)

        self.assertEqual(estimated["detections"][0]["class_name"], "Unknown")
        self.assertNotIn("weight_estimate", estimated["detections"][0])

    def test_attaches_weight_estimate_to_each_camera_detection(self):
        result = {
            "width": 640,
            "height": 480,
            "detections": [
                {"class_name": "pig", "box": {"x": 0, "y": 0, "w": 20, "h": 40}},
                {"class_name": "pig", "box": {"x": 0, "y": 0, "w": 10, "h": 20}},
            ],
        }

        estimated = add_weight_estimates(result)

        self.assertEqual(len(estimated["detections"]), 2)
        self.assertGreater(estimated["detections"][0]["weight_estimate"]["weight_pounds"], 0)
        self.assertLess(
            estimated["detections"][1]["weight_estimate"]["weight_pounds"],
            estimated["detections"][0]["weight_estimate"]["weight_pounds"],
        )

class CameraDetectionApiWeightTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_api_estimates_box_without_setup_fields(self):
        detector_result = {
            "width": 640,
            "height": 480,
            "inference_ms": 1,
            "posture_status": {"available": False, "classes": [], "message": "No weights configured."},
            "detections": [
                {
                    "class_name": "pig",
                    "confidence": 0.9,
                    "box": {"x": 0, "y": 0, "w": 20, "h": 40},
                    "posture": {"label": "standing", "confidence": 0.92},
                }
            ],
        }
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45682))
        with patch("app.pig_detection_service.detect", return_value=detector_result):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/camera/detect",
                    files={"image": ("frame.jpg", b"image", "image/jpeg")},
                )

        self.assertEqual(response.status_code, 200, response.text)
        detection = response.json()["detections"][0]
        self.assertIn("weight_pounds", detection["weight_estimate"])
        self.assertIn("weight_kg", detection["weight_estimate"])
        self.assertEqual(detection["posture"], {"label": "standing", "confidence": 0.92})


if __name__ == "__main__":
    unittest.main()
