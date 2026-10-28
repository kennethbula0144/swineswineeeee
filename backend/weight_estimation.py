import math
from typing import Any

POUNDS_TO_KILOGRAMS = 0.45359237
METERS_TO_INCHES = 39.3700787402
CAMERA_HEIGHT_METERS = 2.0
HORIZONTAL_FOV_DEGREES = 90.0
VERTICAL_FOV_DEGREES = 60.0


def estimate_weight_from_detection(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, float | str]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    box = detection["box"]
    box_width_inches = (
        float(box["w"])
        / 100
        * 2
        * CAMERA_HEIGHT_METERS
        * math.tan(math.radians(HORIZONTAL_FOV_DEGREES) / 2)
        * METERS_TO_INCHES
    )
    box_height_inches = (
        float(box["h"])
        / 100
        * 2
        * CAMERA_HEIGHT_METERS
        * math.tan(math.radians(VERTICAL_FOV_DEGREES) / 2)
        * METERS_TO_INCHES
    )
    girth_inches = min(box_width_inches, box_height_inches)
    body_length_inches = max(box_width_inches, box_height_inches)
    weight_pounds = girth_inches * girth_inches * body_length_inches / 400
    return {
        "girth_proxy_inches": round(girth_inches, 2),
        "body_length_proxy_inches": round(body_length_inches, 2),
        "weight_pounds": round(weight_pounds, 2),
        "weight_kg": round(weight_pounds * POUNDS_TO_KILOGRAMS, 2),
        "method": "overhead_bbox_fixed_camera_proxy",
    }


def add_weight_estimates(
    result: dict[str, Any],
) -> dict[str, Any]:
    result["detections"] = [
        {
            **detection,
            **(
                {
                    "weight_estimate": estimate_weight_from_detection(
                        detection,
                        result["width"],
                        result["height"],
                    )
                }
                if detection.get("class_name") != "Unknown"
                else {}
            ),
        }
        for detection in result["detections"]
    ]
    return result
