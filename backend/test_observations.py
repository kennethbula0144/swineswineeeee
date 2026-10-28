import unittest
from unittest.mock import patch
from uuid import UUID

import httpx

from app import app


PIG_ID = "9e4ad839-f2ed-45d9-9952-9b75b7bd64d2"


class CameraObservationApiTests(unittest.IsolatedAsyncioTestCase):
    async def post_observation(self, body):
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45684))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/observations", json=body)

    async def test_links_identity_updates_weight_and_saves_observation(self):
        pig = {"id": PIG_ID, "ai_identity": "Pig_1", "pen_tag": 3}
        saved = {}

        def create_record(table, values):
            saved.update(table=table, values=values)
            return {"id": "observation-id", **values}

        with (
            patch("app.list_records", return_value=[pig]) as list_records,
            patch("app.create_record", side_effect=create_record),
            patch("app.update_record", return_value=pig) as update_record,
        ):
            response = await self.post_observation(
                {
                    "ai_identity": "Pig 1",
                    "weight_kg": 81.25,
                    "confidence": 0.93,
                    "bounding_box": {"x": 10, "y": 20, "w": 30, "h": 40},
                }
            )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(saved["table"], "observations")
        self.assertEqual(saved["values"]["pig_id"], PIG_ID)
        self.assertEqual(saved["values"]["pen_tag"], 3)
        list_records.assert_called_once_with("pigs", order_by="pen_tag")
        update_record.assert_called_once_with(
            "pigs", PIG_ID, {"registered_weight": 81.25}
        )

    async def test_links_explicit_pig_id_when_identity_is_shared_across_pens(self):
        pig = {"id": PIG_ID, "ai_identity": "Pig_1", "pen_tag": 3}
        saved = {}
        with (
            patch("app.list_records", return_value=[pig]) as list_records,
            patch(
                "app.create_record",
                side_effect=lambda table, values: saved.update(table=table, values=values)
                or {"id": "observation-id", **values},
            ),
            patch("app.update_record", return_value=pig) as update_record,
        ):
            response = await self.post_observation(
                {
                    "pig_id": PIG_ID,
                    "pen_tag": 3,
                    "ai_identity": "Pig_1",
                    "weight_kg": 81.25,
                }
            )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(saved["values"]["pig_id"], PIG_ID)
        self.assertEqual(saved["values"]["pen_tag"], 3)
        list_records.assert_called_once_with(
            "pigs", order_by="pen_tag", filters={"id": PIG_ID}
        )
        update_record.assert_called_once_with(
            "pigs", PIG_ID, {"registered_weight": 81.25}
        )

    async def test_unknown_identity_is_saved_without_linking_or_weight_update(self):
        with (
            patch("app.list_records") as list_records,
            patch(
                "app.create_record",
                return_value={"id": "observation-id", "ai_identity": "Unknown"},
            ) as create_record,
            patch("app.update_record") as update_record,
        ):
            response = await self.post_observation(
                {"ai_identity": "Unknown", "confidence": 0.8}
            )

        self.assertEqual(response.status_code, 201, response.text)
        list_records.assert_not_called()
        create_record.assert_called_once()
        update_record.assert_not_called()

    async def test_saves_thermal_surface_temperature_on_observation(self):
        saved = {}
        with (
            patch("app.list_records") as list_records,
            patch(
                "app.create_record",
                side_effect=lambda table, values: saved.update(table=table, values=values)
                or {"id": "observation-id", **values},
            ),
            patch("app.update_record") as update_record,
        ):
            response = await self.post_observation(
                {
                    "ai_identity": "Unknown",
                    "thermal_surface_c": 34.6,
                }
            )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(saved["values"]["thermal_surface_c"], 34.6)
        list_records.assert_not_called()
        update_record.assert_not_called()

    async def test_duplicate_registry_identity_rejects_ambiguous_link(self):
        pigs = [
            {"id": PIG_ID, "ai_identity": "Pig_1", "pen_tag": 3},
            {"id": "a01c2df2-65a2-4ee0-aad0-c6abc4a3555d", "ai_identity": "Pig 1", "pen_tag": 4},
        ]
        with (
            patch("app.list_records", return_value=pigs),
            patch("app.create_record") as create_record,
            patch("app.update_record") as update_record,
        ):
            response = await self.post_observation(
                {"ai_identity": "Pig 1", "weight_kg": 81.25}
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("multiple registry entries", response.json()["detail"])
        create_record.assert_not_called()
        update_record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
