import unittest
import urllib.error
from unittest.mock import MagicMock, patch

import httpx

from app import app


class TapoWebRTCTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45685))
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def test_forwards_sdp_offer_to_fixed_go2rtc_tapo_stream(self):
        answer_sdp = "v=0\r\no=go2rtc 0 0 IN IP4 127.0.0.1\r\n"
        response = MagicMock()
        response.__enter__.return_value.read.return_value = answer_sdp.encode("utf-8")

        with patch("app.urllib.request.urlopen", return_value=response) as urlopen:
            result = await self.client.post(
                "/api/camera/tapo-c230/offer",
                json={"sdp": "v=0\r\no=browser 0 0 IN IP4 127.0.0.1\r\n"},
            )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {"type": "answer", "sdp": answer_sdp})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:1984/api/webrtc?src=tapo_c230")
        self.assertEqual(request.get_header("Content-type"), "application/sdp")
        self.assertEqual(request.data, b"v=0\r\no=browser 0 0 IN IP4 127.0.0.1\r\n")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 15)

    async def test_reports_go2rtc_unavailable_without_exposing_upstream_error(self):
        with patch(
            "app.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = await self.client.post(
                "/api/camera/tapo-c230/offer",
                json={"sdp": "v=0\r\no=browser 0 0 IN IP4 127.0.0.1\r\n"},
            )

        self.assertEqual(result.status_code, 503)
        self.assertEqual(
            result.json()["detail"],
            "Could not reach go2rtc. Make sure the local go2rtc service is running.",
        )

    async def test_rejects_invalid_sdp_before_contacting_go2rtc(self):
        with patch("app.urllib.request.urlopen") as urlopen:
            result = await self.client.post(
                "/api/camera/tapo-c230/offer",
                json={"sdp": "invalid offer"},
            )

        self.assertEqual(result.status_code, 422)
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
