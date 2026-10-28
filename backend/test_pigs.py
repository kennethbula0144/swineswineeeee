import unittest
from unittest.mock import patch

import httpx

from app import app


class PigRegistryApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_pig_accepts_pen_tag_above_previous_limit(self):
        created = {}

        def create_record(table, values):
            created.update(table=table, values=values)
            return {"id": "pig-id", **values}

        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45685))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with (
                patch("app.count_records", return_value=0),
                patch("app.list_records", return_value=[]),
                patch("app.create_record", side_effect=create_record),
            ):
                response = await client.post(
                    "/api/pigs",
                    json={
                        "ai_identity": "Pig_1",
                        "pen_tag": 21,
                        "growth_stage": "Grower",
                        "registered_weight": 35.5,
                    },
                )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["pen_tag"], 21)
        self.assertEqual(created["table"], "pigs")
        self.assertEqual(created["values"]["pen_tag"], 21)

    async def test_create_pig_retry_returns_existing_client_generated_id(self):
        pig_id = "9e4ad839-f2ed-45d9-9952-9b75b7bd64d2"
        existing = {
            "id": pig_id,
            "ai_identity": "Pig_1",
            "pen_tag": 1,
            "growth_stage": "Grower",
            "registered_weight": 35.5,
            "status": "Active",
            "notes": "",
        }
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45690))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with (
                patch("app.list_records", return_value=[existing]) as list_records,
                patch("app.count_records") as count_records,
                patch("app.create_record") as create_record,
            ):
                response = await client.post(
                    "/api/pigs",
                    json={
                        "id": pig_id,
                        "ai_identity": "Pig_1",
                        "pen_tag": 1,
                        "growth_stage": "Grower",
                        "registered_weight": 35.5,
                    },
                )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["id"], pig_id)
        list_records.assert_called_once_with(
            "pigs", order_by="pen_tag", filters={"id": pig_id}
        )
        count_records.assert_not_called()
        create_record.assert_not_called()

    async def test_create_pig_can_share_a_pen_with_existing_pigs(self):
        created = {}
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45687))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with (
                patch("app.count_records", return_value=1),
                patch("app.list_records", return_value=[]),
                patch(
                    "app.create_record",
                    side_effect=lambda table, values: created.update(
                        table=table, values=values
                    )
                    or {"id": "pig-id", **values},
                ),
            ):
                response = await client.post(
                    "/api/pigs",
                    json={
                        "ai_identity": "Pig_2",
                        "pen_tag": 1,
                        "growth_stage": "Grower",
                        "registered_weight": 35.5,
                    },
                )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["pen_tag"], 1)
        self.assertEqual(created["values"]["pen_tag"], 1)

    async def test_create_pig_allows_same_identity_in_a_different_pen(self):
        created = {}
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45688))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with (
                patch("app.count_records", return_value=1),
                patch("app.list_records", return_value=[]) as list_records,
                patch(
                    "app.create_record",
                    side_effect=lambda table, values: created.update(
                        table=table, values=values
                    )
                    or {"id": "pig-id", **values},
                ),
            ):
                response = await client.post(
                    "/api/pigs",
                    json={
                        "ai_identity": "Pig_1",
                        "pen_tag": 2,
                        "growth_stage": "Grower",
                        "registered_weight": 35.5,
                    },
                )

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(created["values"]["pen_tag"], 2)
        list_records.assert_called_once_with(
            "pigs", order_by="pen_tag", filters={"pen_tag": 2}
        )

    async def test_create_pig_rejects_same_identity_in_same_pen(self):
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45689))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with (
                patch("app.count_records", return_value=1),
                patch(
                    "app.list_records",
                    return_value=[
                        {"ai_identity": "pig 1", "pen_tag": 1},
                    ],
                ),
                patch("app.create_record") as create_record,
            ):
                response = await client.post(
                    "/api/pigs",
                    json={
                        "ai_identity": "Pig_1",
                        "pen_tag": 1,
                        "growth_stage": "Grower",
                        "registered_weight": 35.5,
                    },
                )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("already assigned to pen 01", response.json()["detail"])
        create_record.assert_not_called()

    async def test_create_pig_rejects_a_pen_that_already_has_twenty_pigs(self):
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45686))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with (
                patch("app.count_records", return_value=20) as count_records,
                patch("app.list_records", return_value=[]),
                patch("app.create_record") as create_record,
            ):
                response = await client.post(
                    "/api/pigs",
                    json={
                        "ai_identity": "Pig_21",
                        "pen_tag": 1,
                        "growth_stage": "Grower",
                        "registered_weight": 35.5,
                    },
                )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("limit of 20 pigs", response.json()["detail"])
        count_records.assert_called_once_with("pigs", pen_tag=1)
        create_record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
