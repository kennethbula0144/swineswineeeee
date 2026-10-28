import unittest
from unittest.mock import patch

import httpx

from app import app
from thermal import THERMAL_FRAME_MAX_AGE_SECONDS, THERMAL_PIXEL_COUNT, ThermalFrameService


class ThermalFrameServiceTests(unittest.TestCase):
    def test_status_includes_current_pixels_for_thermal_preview(self):
        service = ThermalFrameService()
        pixels = [30.0] * THERMAL_PIXEL_COUNT
        service.ingest(pixels, 23.5)

        status = service.status()

        self.assertTrue(status["available"])
        self.assertEqual(status["pixels"], pixels)

    def test_maps_hottest_quarter_of_pig_region_to_apparent_surface_temperature(self):
        service = ThermalFrameService()
        pixels = [25.0] * THERMAL_PIXEL_COUNT
        pixels[10 * 32 + 14] = 34.0
        pixels[10 * 32 + 15] = 36.0
        pixels[10 * 32 + 16] = 38.0
        pixels[11 * 32 + 14] = 40.0
        pixels[11 * 32 + 15] = 40.0
        pixels[11 * 32 + 16] = 40.0
        pixels[12 * 32 + 14] = 40.0
        pixels[12 * 32 + 15] = 40.0
        pixels[12 * 32 + 16] = 40.0
        service.ingest(pixels, 23.5)

        detections = [{"box": {"x": 40, "y": 40, "w": 20, "h": 20}}]
        result = service.annotate_detections(detections)

        self.assertEqual(result[0]["surface_temperature_c"], 40.0)

    def test_does_not_annotate_detections_without_a_recent_sensor_frame(self):
        service = ThermalFrameService()
        detection = {"box": {"x": 40, "y": 40, "w": 20, "h": 20}}
        service.ingest([30.0] * THERMAL_PIXEL_COUNT, 22.0)

        with patch(
            "thermal.time.time",
            return_value=service._updated_at + THERMAL_FRAME_MAX_AGE_SECONDS + 1,
        ):
            result = service.annotate_detections([detection])

        self.assertNotIn("surface_temperature_c", result[0])

    def test_rejects_non_finite_pixel_values(self):
        service = ThermalFrameService()
        pixels = [25.0] * THERMAL_PIXEL_COUNT
        pixels[0] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite temperatures"):
            service.ingest(pixels, 22.0)


class ThermalFrameApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        transport = httpx.ASGITransport(app=app, client=("192.168.1.55", 45686))
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_accepts_authenticated_esp32_frame_from_lan(self):
        token = "test-thermal-token-that-is-at-least-32-chars"
        with patch.dict("os.environ", {"ESP32_THERMAL_TOKEN": token}):
            response = await self.client.post(
                "/api/thermal/frame",
                headers={"X-Thermal-Token": token},
                json={"pixels": [30.0] * THERMAL_PIXEL_COUNT, "ambient_c": 24.0},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["received"])
        self.assertTrue(response.json()["available"])

    async def test_rejects_frame_without_device_token(self):
        token = "test-thermal-token-that-is-at-least-32-chars"
        with patch.dict("os.environ", {"ESP32_THERMAL_TOKEN": token}):
            response = await self.client.post(
                "/api/thermal/frame",
                json={"pixels": [30.0] * THERMAL_PIXEL_COUNT, "ambient_c": 24.0},
            )

        self.assertEqual(response.status_code, 401)

    async def test_rejects_frame_with_wrong_pixel_count(self):
        token = "test-thermal-token-that-is-at-least-32-chars"
        with patch.dict("os.environ", {"ESP32_THERMAL_TOKEN": token}):
            response = await self.client.post(
                "/api/thermal/frame",
                headers={"X-Thermal-Token": token},
                json={"pixels": [30.0] * 10, "ambient_c": 24.0},
            )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
