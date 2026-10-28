import ipaddress
import hmac
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from weight_estimation import add_weight_estimates
from posture import posture_detection_service
from posture_training import PostureTrainingError, posture_training_service
from detection import (
    MAX_FRAME_BYTES,
    DetectorUnavailableError,
    pig_detection_service,
)
from datasets import (
    DatasetArchiveError,
    DatasetNotFoundError,
    DatasetSizeError,
    list_uploaded_datasets,
    save_dataset_archive,
)
from storage import (
    ConfigurationError,
    SupabaseOperationError,
    count_records,
    create_record,
    delete_record,
    get_client,
    list_records,
    update_record,
)
from training import DatasetTrainingError, dataset_training_service
from thermal import THERMAL_PIXEL_COUNT, thermal_frame_service

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

def require_local_client(request: Request) -> None:
    if request.url.path == "/api/thermal/frame":
        return
    proxy_headers = {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
    }
    if proxy_headers.intersection(request.headers.keys()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Passwordless database access is restricted to direct local connections.",
        )
    if request.client is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Passwordless database access is restricted to this machine.",
        )
    try:
        client_ip = ipaddress.ip_address(request.client.host)
        if isinstance(client_ip, ipaddress.IPv6Address) and client_ip.ipv4_mapped:
            client_ip = client_ip.ipv4_mapped
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Passwordless database access is restricted to this machine.",
        ) from exc
    if not client_ip.is_loopback:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Passwordless database access is restricted to this machine.",
        )


app = FastAPI(
    title="Smart Swine API",
    version="1.0.0",
    dependencies=[Depends(require_local_client)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


class PigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID | None = None
    ai_identity: str | None = Field(default=None, max_length=50)
    pen_tag: int = Field(ge=1, le=2_147_483_647)
    growth_stage: str = Field(min_length=1, max_length=40)
    registered_weight: float = Field(ge=0)
    status: str = Field(default="Active", pattern="^(Active|Quarantined|Inactive)$")
    notes: str = Field(default="", max_length=1000)


class FeedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID | None = None
    pig_id: UUID
    pen_tag: int = Field(ge=1, le=2_147_483_647)
    feed_type: str = Field(min_length=1, max_length=120)
    quantity_kg: float = Field(gt=0)
    cost_per_kg: float = Field(ge=0)
    fed_on: date


class AlertUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern="^(Pending Review|Referred to Vet|Resolved)$")


class AlertInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID | None = None
    pig_id: UUID | None = None
    ai_identity: str | None = Field(default=None, max_length=50)
    pen_tag: int | None = Field(default=None, ge=1, le=2_147_483_647)
    issue: str = Field(min_length=1, max_length=200)
    severity: str = Field(default="Medium", pattern="^(Low|Medium|High|Critical)$")
    status: str = Field(default="Pending Review", pattern="^(Pending Review|Referred to Vet|Resolved)$")
    details: str = Field(default="", max_length=2000)


class ObservationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID | None = None
    pig_id: UUID | None = None
    ai_identity: str | None = Field(default=None, max_length=50)
    pen_tag: int | None = Field(default=None, ge=1, le=2_147_483_647)
    weight_kg: float | None = Field(default=None, ge=0)
    thermal_surface_c: float | None = Field(default=None, ge=-40, le=300)
    behavior: str | None = Field(default=None, max_length=60)
    confidence: float | None = Field(default=None, ge=0, le=1)
    bounding_box: dict[str, float] | None = None


class WebRTCOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sdp: str = Field(min_length=1, max_length=65536)


class ThermalFrameInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pixels: list[float] = Field(min_length=THERMAL_PIXEL_COUNT, max_length=THERMAL_PIXEL_COUNT)
    ambient_c: float = Field(ge=-40, le=125)


def require_thermal_device_token(request: Request) -> None:
    expected_token = os.getenv("ESP32_THERMAL_TOKEN", "")
    supplied_token = request.headers.get("x-thermal-token", "")
    if len(expected_token) < 32 or not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thermal sensor authentication failed. Configure the ESP32 thermal token.",
        )


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(_, exc: ConfigurationError):
    return _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.exception_handler(SupabaseOperationError)
async def database_error_handler(_, exc: SupabaseOperationError):
    logger.error("Supabase operation failed: %s", exc)
    return _http_error(status.HTTP_502_BAD_GATEWAY, str(exc))


def _http_error(code: int, message: str) -> Response:
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=code, content={"detail": message})


@app.get("/api/health")
def health() -> dict[str, str]:
    try:
        get_client().table("pigs").select("id").limit(1).execute()
    except ConfigurationError:
        raise
    except Exception as exc:
        logger.exception("Supabase database health check failed.")
        if "Invalid API key" in str(exc):
            message = (
                "Supabase rejected SUPABASE_SECRET_KEY. Copy the current secret key "
                "from Supabase Settings > API Keys into the project .env file, then restart the API."
            )
        else:
            message = (
                "Supabase database check failed. Verify the project URL and apply "
                "supabase/migrations/001_initial_schema.sql if the tables are missing."
            )
        raise SupabaseOperationError(message) from exc
    return {"status": "ok", "database": "connected"}


@app.get("/api/camera/status")
def camera_status() -> dict[str, Any]:
    detector_status = pig_detection_service.status()
    detector_status["posture"] = posture_detection_service.status()
    return detector_status


@app.get("/api/thermal/status")
def thermal_status() -> dict[str, Any]:
    return thermal_frame_service.status()


@app.post("/api/thermal/frame", dependencies=[Depends(require_thermal_device_token)])
def receive_thermal_frame(frame: ThermalFrameInput) -> dict[str, Any]:
    try:
        thermal_frame_service.ingest(frame.pixels, frame.ambient_c)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return {"received": True, **thermal_frame_service.status()}


def _request_tapo_webrtc_answer(sdp: str) -> str:
    request = urllib.request.Request(
        "http://127.0.0.1:1984/api/webrtc?src=tapo_c230",
        data=sdp.encode("utf-8"),
        headers={"Content-Type": "application/sdp"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            answer = response.read(65537)
    except urllib.error.HTTPError as exc:
        logger.warning("go2rtc rejected a Tapo C230 WebRTC offer (HTTP %s).", exc.code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="go2rtc could not start the Tapo stream. Check the RTSP stream and camera network connection.",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Could not connect to go2rtc for the Tapo stream (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach go2rtc. Make sure the local go2rtc service is running.",
        ) from exc

    if len(answer) > 65536:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="go2rtc returned an oversized WebRTC answer.",
        )
    try:
        answer_sdp = answer.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="go2rtc returned an invalid WebRTC answer.",
        ) from exc
    if not answer_sdp.startswith("v=0"):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="go2rtc returned an invalid WebRTC answer.",
        )
    return answer_sdp


@app.post("/api/camera/tapo-c230/offer")
async def tapo_webrtc_offer(offer: WebRTCOffer) -> dict[str, str]:
    if not offer.sdp.startswith("v=0"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A valid WebRTC SDP offer is required.",
        )
    answer = await run_in_threadpool(_request_tapo_webrtc_answer, offer.sdp)
    return {"type": "answer", "sdp": answer}


@app.post("/api/camera/detect")
async def detect_camera_frame(
    image: UploadFile = File(...),
) -> dict[str, Any]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG, or WebP image frame.",
        )
    frame_bytes = await image.read(MAX_FRAME_BYTES + 1)
    if len(frame_bytes) > MAX_FRAME_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image is too large. Frames must be 8 MB or smaller.",
        )
    try:
        result = await run_in_threadpool(pig_detection_service.detect, frame_bytes)
        result = add_weight_estimates(result)
        result["detections"] = thermal_frame_service.annotate_detections(result["detections"])
        return result
    except DetectorUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Pig detection failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pig detection failed. Check the backend logs.",
        ) from exc


@app.get("/api/datasets")
def get_datasets() -> list[dict[str, Any]]:
    return list_uploaded_datasets()


@app.post("/api/datasets", status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        return await run_in_threadpool(save_dataset_archive, file.file, file.filename or "")
    except DatasetSizeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except DatasetArchiveError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()


@app.delete("/api/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_dataset(dataset_id: str) -> Response:
    posture_status = posture_training_service.status()
    if posture_status["status"] == "running" and posture_status["dataset_id"] == dataset_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This dataset cannot be deleted while its posture training job is running.",
        )
    try:
        dataset_training_service.delete_dataset(dataset_id)
    except DatasetTrainingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/training/status")
def get_training_status() -> dict[str, Any]:
    return dataset_training_service.status()


@app.post("/api/datasets/{dataset_id}/train", status_code=status.HTTP_202_ACCEPTED)
def train_dataset(dataset_id: str) -> dict[str, Any]:
    if posture_training_service.status()["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A posture training job is already running.",
        )
    try:
        return dataset_training_service.start(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetTrainingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.get("/api/posture-training/status")
def get_posture_training_status() -> dict[str, Any]:
    return posture_training_service.status()


@app.post("/api/datasets/{dataset_id}/train-posture", status_code=status.HTTP_202_ACCEPTED)
def train_posture_dataset(dataset_id: str) -> dict[str, Any]:
    if dataset_training_service.status()["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A pig detector training job is already running.",
        )
    try:
        return posture_training_service.start(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PostureTrainingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.get("/api/pigs")
def get_pigs() -> list[dict[str, Any]]:
    return list_records("pigs", order_by="pen_tag")


@app.post("/api/pigs", status_code=status.HTTP_201_CREATED)
def add_pig(pig: PigInput) -> dict[str, Any]:
    if pig.id is not None:
        existing_pigs = list_records("pigs", order_by="pen_tag", filters={"id": str(pig.id)})
        if existing_pigs:
            return existing_pigs[0]
    identity_key = _identity_key(pig.ai_identity)
    if identity_key and identity_key != "unknown":
        pen_pigs = list_records(
            "pigs",
            order_by="pen_tag",
            filters={"pen_tag": pig.pen_tag},
        )
        if any(_identity_key(existing.get("ai_identity")) == identity_key for existing in pen_pigs):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"AI identity {pig.ai_identity!r} is already assigned "
                    f"to pen {pig.pen_tag:02d}."
                ),
            )
    if count_records("pigs", pen_tag=pig.pen_tag) >= 20:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pen {pig.pen_tag:02d} has reached its limit of 20 pigs.",
        )
    return create_record(
        "pigs",
        {
            **pig.model_dump(mode="json", exclude_none=True),
            "pen_tag": pig.pen_tag,
        },
    )


@app.delete("/api/pigs/{pig_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_pig(pig_id: UUID) -> Response:
    delete_record("pigs", str(pig_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/observations")
def get_observations() -> list[dict[str, Any]]:
    return list_records("observations", order_by="created_at", descending=True, limit=100)


def _identity_key(identity: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (identity or "").lower())


@app.post("/api/observations", status_code=status.HTTP_201_CREATED)
def add_observation(observation: ObservationInput) -> dict[str, Any]:
    values = observation.model_dump(mode="json", exclude_none=True)
    matched_pig = None
    identity_key = _identity_key(observation.ai_identity)
    if observation.pig_id is not None:
        matches = list_records(
            "pigs",
            order_by="pen_tag",
            filters={"id": str(observation.pig_id)},
        )
        if not matches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The observation references a pig that does not exist.",
            )
        matched_pig = matches[0]
    elif identity_key and identity_key != "unknown":
        matches = [
            pig
            for pig in list_records("pigs", order_by="pen_tag")
            if _identity_key(pig.get("ai_identity")) == identity_key
        ]
        if len(matches) > 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"AI identity {observation.ai_identity!r} is assigned to multiple "
                    "registry entries. Resolve the duplicate identity before saving."
                ),
            )
        if matches:
            matched_pig = matches[0]
            values["pig_id"] = matched_pig["id"]
            values["pen_tag"] = matched_pig["pen_tag"]

    created = create_record("observations", values)
    if matched_pig is not None and observation.weight_kg is not None:
        update_record(
            "pigs",
            str(matched_pig["id"]),
            {"registered_weight": observation.weight_kg},
        )
    return created


@app.get("/api/alerts")
def get_alerts() -> list[dict[str, Any]]:
    return list_records("alerts", order_by="created_at", descending=True, limit=100)


@app.post("/api/alerts", status_code=status.HTTP_201_CREATED)
def add_alert(alert: AlertInput) -> dict[str, Any]:
    return create_record("alerts", alert.model_dump(mode="json", exclude_none=True))


@app.patch("/api/alerts/{alert_id}")
def change_alert_status(alert_id: UUID, update: AlertUpdate) -> dict[str, Any]:
    return update_record("alerts", str(alert_id), update.model_dump())


@app.get("/api/feed-logs")
def get_feed_logs() -> list[dict[str, Any]]:
    return list_records("feed_logs", order_by="fed_on", descending=True, limit=500)


@app.post("/api/feed-logs", status_code=status.HTTP_201_CREATED)
def add_feed_log(feed: FeedInput) -> dict[str, Any]:
    return create_record("feed_logs", feed.model_dump(mode="json", exclude_none=True))


@app.delete("/api/feed-logs/{feed_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_feed_log(feed_id: UUID) -> Response:
    delete_record("feed_logs", str(feed_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
