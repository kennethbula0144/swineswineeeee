import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from detection import INFERENCE_IMAGE_SIZE, MAX_DETECTIONS, PigDetectionService


class PigDetectionServiceTests(unittest.TestCase):
    def test_returns_at_most_four_detections_ordered_by_confidence(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded, frame_bytes = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        boxes = [
            SimpleNamespace(
                cls=np.array([0]),
                conf=np.array([confidence]),
                xyxy=np.array([[index, index, index + 10, index + 10]]),
            )
            for index, confidence in enumerate((0.3, 0.95, 0.6, 0.8, 0.7, 0.4))
        ]
        model = SimpleNamespace(
            names={0: "pig"},
            predict=lambda **_kwargs: [
                SimpleNamespace(names={0: "pig"}, boxes=boxes)
            ],
        )
        service = PigDetectionService()

        with (
            patch.object(service, "_load_model", return_value=model),
            patch("detection.posture_detection_service.classify_detections", side_effect=lambda _frame, detections: {
                "detections": detections,
                "posture_status": {"available": False, "classes": [], "message": None},
            }),
        ):
            result = service.detect(frame_bytes.tobytes())

        confidences = [detection["confidence"] for detection in result["detections"]]
        self.assertEqual(len(confidences), MAX_DETECTIONS)
        self.assertEqual(confidences, [0.95, 0.8, 0.7, 0.6])

    def test_detects_individual_pig_classes(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded, frame_bytes = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        names = {
            0: "pig-monitoring",
            1: "Pig 1",
            2: "Pig 2",
            3: "Pig 3",
            4: "Pig 4",
            5: "batch",
        }
        boxes = [
            SimpleNamespace(
                cls=np.array([class_id]),
                conf=np.array([0.9]),
                xyxy=np.array([[index, index, index + 10, index + 10]]),
            )
            for index, class_id in enumerate((1, 2, 3, 4, 5))
        ]
        predict = unittest.mock.Mock(
            return_value=[SimpleNamespace(names=names, boxes=boxes)]
        )
        model = SimpleNamespace(names=names, predict=predict)
        service = PigDetectionService()

        with (
            patch.object(service, "_load_model", return_value=model),
            patch("detection.posture_detection_service.classify_detections", side_effect=lambda _frame, detections: {
                "detections": detections,
                "posture_status": {"available": False, "classes": [], "message": None},
            }),
        ):
            result = service.detect(frame_bytes.tobytes())

        self.assertEqual(predict.call_args.kwargs["classes"], [1, 2, 3, 4])
        self.assertEqual(
            [detection["class_name"] for detection in result["detections"]],
            ["Pig 1", "Pig 2", "Pig 3", "Pig 4"],
        )

    def test_filters_non_pig_detector_classes(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded, frame_bytes = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        names = {0: "Pig 1", 1: "batch"}
        boxes = [
            SimpleNamespace(cls=np.array([0]), conf=np.array([0.9]), xyxy=np.array([[0, 0, 20, 20]])),
            SimpleNamespace(cls=np.array([1]), conf=np.array([0.8]), xyxy=np.array([[30, 30, 60, 60]])),
        ]
        model = SimpleNamespace(
            names=names,
            predict=lambda **_kwargs: [SimpleNamespace(names=names, boxes=boxes)],
        )
        service = PigDetectionService()

        with (
            patch.object(service, "_load_model", return_value=model),
            patch(
                "detection.posture_detection_service.classify_detections",
                side_effect=lambda _frame, detections: {
                    "detections": detections,
                    "posture_status": {"available": False, "classes": [], "message": None},
                },
            ),
        ):
            result = service.detect(frame_bytes.tobytes())

        self.assertEqual(
            [detection["class_name"] for detection in result["detections"]],
            ["Pig 1"],
        )

    def test_requests_high_resolution_inference_and_four_detection_limit(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded, frame_bytes = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        predict = unittest.mock.Mock(return_value=[])
        model = SimpleNamespace(names={0: "pig"}, predict=predict)
        service = PigDetectionService()

        with (
            patch.object(service, "_load_model", return_value=model),
            patch("detection.posture_detection_service.classify_detections", return_value={
                "detections": [],
                "posture_status": {"available": False, "classes": [], "message": None},
            }),
        ):
            service.detect(frame_bytes.tobytes())

        self.assertEqual(predict.call_args.kwargs["imgsz"], INFERENCE_IMAGE_SIZE)
        self.assertEqual(predict.call_args.kwargs["max_det"], MAX_DETECTIONS)
        self.assertTrue(predict.call_args.kwargs["agnostic_nms"])


if __name__ == "__main__":
    unittest.main()
