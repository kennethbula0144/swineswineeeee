# Smart Swine

Smart Swine is a starter workspace for a pig-farm monitoring dashboard, model-training pipeline, and FastAPI service.

## Connect Supabase

1. Create a Supabase project.
2. Run the SQL files in [`supabase/migrations`](./supabase/migrations) in numeric order. The migrations create the pigs, observations, alerts, and feed log tables, expand the pen-tag range, and limit each pen to 20 pigs. They do not insert sample records.
3. The dashboard does not use Supabase Auth or create a public profile table. Supabase's managed `auth` schema is left intact.
4. Copy `.env.example` to `.env` and set the Supabase project URL and a valid **secret** key from Supabase **Settings → API Keys**. FastAPI uses that key only; never place it in a `VITE_*` variable or frontend file. `SUPABASE_ANON_KEY` and the publishable key are not used for privileged database operations.
5. Install and start the backend:

   ```powershell
   cd backend
   python -m pip install -r requirements.txt
   uvicorn app:app --reload --host 127.0.0.1 --port 8765
   ```

6. In another terminal, start the dashboard:

   ```powershell
   cd smartswine-dashboard
   npm install
   npm run dev
   ```

The dashboard shows the Smart Swine loading screen, then connects to Supabase through FastAPI without a Supabase sign-in. By default, the backend is loopback-only and passwordless farm-data access is restricted to requests originating from the same machine. The optional ESP32 thermal feature requires binding the backend to the private LAN; only its token-protected frame-upload endpoint accepts LAN traffic. Never expose the backend to the public internet or through a public reverse proxy. The Supabase secret remains on the backend and is never sent to the browser. The Vite config loads the API URL from the project-root `.env`; restart Vite after changing it. Use `npm run build` from `smartswine-dashboard` for a production build.

The production dashboard caches its app shell and stores pig, observation, alert, and feed records in browser IndexedDB. These records can be viewed and changed offline; additions, deletions, and alert status changes are queued and automatically retried when the dashboard can reach the backend and Supabase. Local pending changes take precedence during reconciliation. Keep the dashboard open or install it as a web app after loading it once so the cached shell is available offline. Service workers require HTTPS or localhost. Camera detection still requires the local FastAPI backend and detector; datasets and training also require the backend. Browser storage is device- and browser-specific, so it is not a substitute for regular backups or a shared database.

The database starts empty. Row-level security is enabled and no browser-role policies are added; only the local FastAPI backend uses its private secret key. The live monitor can select an available browser camera and runs the pig-specific YOLO detector in the local backend. Camera inference uses 960-pixel frames, keeps only pig-identity classes (`pig`, `Pig 1` through `Pig 4`), and is capped at the four highest-confidence pig detections per frame. Nearly pure-black camera frames are flagged in the preview and skipped to avoid false detections when the camera shutter or privacy mode blocks the image. The detector uses learned image features rather than a hand-written color threshold, but it can still mistake background objects for pigs if training data teaches it that visual association. To reduce these false positives, include representative pen-background images with no pigs and no bounding-box labels as negative examples, plus accurately labeled pigs from the actual camera angle and lighting; retrain and verify against separate held-out footage. Increasing confidence alone may also hide real pigs, so test any threshold change against both false-positive and missed-pig examples. Higher-resolution inference can improve small-pig detail but may take longer. Each five-second live scan automatically saves returned pig detections, identity, posture, confidence, and bounding box to Supabase observations. When a detected identity uniquely matches one pig in the registry (spaces and underscores are treated equivalently), the observation is linked to that pig, and its latest camera weight estimate replaces the registry weight. Duplicate registry identities are rejected for linking so the wrong pig is not updated. Camera access requires browser permission and localhost or HTTPS. Place `pig_detector.pt` in `backend/models` (the local model file is gitignored), or set `PIG_DETECTOR_MODEL` to its path. Use **Settings & training → Train detector** with representative, correctly labeled pig images to fine-tune the object detector; training results vary with dataset quality and should be checked against held-out images. Posture recognition runs after pig detection using a separate YOLO image classifier on each detected pig crop; it refines each detection but does not locate pigs on its own. To activate it, upload a ZIP containing labeled classification folders with matching `train` and `val` splits, such as `train/standing/`, `train/lateral_lying/`, `val/standing/`, and `val/lateral_lying/`; include at least two classes and labeled images in each split/class folder. In **Settings & training**, choose **Train posture** for that dataset. Training uses the YOLO11n classification checkpoint (downloaded by Ultralytics on first use), reports progress, and activates the resulting `pig_posture.pt` model. The existing object-detection dataset has identity labels but no posture labels, so it cannot train posture recognition by itself. Identity and posture labels are shown in the live view and saved in observations. Still-image classification cannot assess walking gait, limping, tail motion, or diagnose illness.

Relative `PIG_DETECTOR_MODEL` and `PIG_POSTURE_MODEL` paths in the project-root `.env` are resolved from the repository root, regardless of the directory used to start FastAPI.

To use a Tapo C230 through the locally running go2rtc service, set the `tapo_c230` stream in `%LOCALAPPDATA%\go2rtc\go2rtc.yaml` to the camera's RTSP URL, replacing the example username, password, and IP address, then restart go2rtc. In the live monitor's **Camera source** menu, select **Tapo C230 (go2rtc)** to connect using WebRTC. Its video is processed by the same in-dashboard pig detection, posture, weight estimation, and automatic observation-saving flow as a browser camera. The local FastAPI backend proxies WebRTC signaling to go2rtc; keep go2rtc bound to localhost. The camera and PC must be able to reach each other over the local network, and the go2rtc WebRTC UDP port (8555 by default) must be available.

### MLX90640-D110 thermal surface screening

The optional [MLX90640-D110](https://www.waveshare.com/wiki/MLX90640-D110_Thermal_Camera) and ESP32-S3 send a 32×24 thermal array to FastAPI over the farm Wi-Fi. The backend maps each detected RGB pig box to the corresponding normalized thermal pixels and attaches an apparent surface-temperature estimate to the live detection and saved observation. The D110 has a 110°×75° field of view; at 2 m, one thermal pixel covers roughly 18×13 cm. This is coarse screening only, not accurate core/body temperature, veterinary diagnosis, or an individual-pig measurement guarantee. Co-mount the sensors, keep their views aligned, and lock the Tapo pan/tilt position. Validate alignment with a known warm target before relying on pig-to-thermal associations. Use the narrower D55 lens or move closer if individual pigs occupy only a few thermal pixels.

Use **MLX90640 thermal view** in **Live pen camera** to display the sensor's latest 32×24 frame as a color heatmap. The color range auto-scales to each fresh frame; the legend shows the frame minimum and maximum, and the optional ambient reading is shown alongside it. The view requires recent authenticated ESP32 uploads and is for visual surface screening, not body/core temperature measurement.

#### Configure the backend for ESP32 access

The default backend bind address is loopback-only. For the ESP32, set a private-LAN bind address and a long random shared token. From the project root, generate a token:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add the generated value to the project-root `.env` as `ESP32_THERMAL_TOKEN=...`. Keep it private; do not commit it or put it in a frontend `VITE_*` setting. Find the PC's Wi-Fi IPv4 address with `ipconfig`, put it in the Arduino sketch's `BACKEND_URL`, and set the same token in `THERMAL_TOKEN`. Bind FastAPI to the LAN interface so the ESP32 can connect:

```powershell
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8765
```

Allow Python/Uvicorn on **Private networks only** in Windows Firewall. The other API routes still reject non-local clients; only the thermal upload endpoint accepts LAN traffic, and it requires the shared token. Do not expose port 8765 to the public internet. The browser dashboard continues to use `http://127.0.0.1:8765`.

#### Flash the ESP32-S3 in Arduino IDE

Open [`firmware/esp32_s3_mlx90640/esp32_s3_mlx90640.ino`](./firmware/esp32_s3_mlx90640/esp32_s3_mlx90640.ino). In Arduino IDE Boards Manager install **esp32 by Espressif Systems**; in Library Manager install **Adafruit MLX90640** (and its **Adafruit BusIO** dependency). Select your ESP32-S3 board and port. Wire sensor **VCC → 3V3**, **GND → GND**, **SDA → GPIO 8**, **SCL → GPIO 9**; change `I2C_SDA_PIN` and `I2C_SCL_PIN` to match the selected board's pinout. Set the Wi-Fi name/password, backend URL (for example `http://192.168.1.25:8765/api/thermal/frame`), and the shared thermal token in the sketch, then upload and check Serial Monitor at 115200 baud. The sketch transmits 32×24 temperature pixels about once per second; the dashboard reports whether a recent thermal frame arrived.

For Python still-image inference, run from the `backend` directory:

```python
from pig_monitoring import detect_pigs

result = detect_pigs("pen.jpg")
for detection in result.detections:
    print(
        detection.class_name,
        detection.confidence,
        detection.x,
        detection.y,
        detection.width,
        detection.height,
    )
```

Bounding-box coordinates and dimensions are pixels relative to the input image. Run the smoke test with `python -m unittest test_smoke -v`.

The live monitor estimates weight automatically from each detected bounding box and displays estimates in kilograms; no calibration or FOV setup is required. For the fixed overhead camera at 2 m, the estimate uses a built-in assumed 90° horizontal × 60° vertical camera view to convert box dimensions to approximate inches. The shorter side is only a rough girth proxy (not true pig circumference); the longer side is used as body length. The app applies `(girth_inches × girth_inches × body_length_inches) / 400` to estimate pounds internally, converts the result to kilograms, and stores kilograms in the observation. These estimates are experimental and must not be treated as scale measurements; actual lens view, camera angle, and animal posture can significantly affect accuracy.

To add training data, open **Settings & training → Dataset uploads**, choose a `.zip` dataset archive, and upload it. Archives are stored locally and extracted beneath `aitrain/datasets`; uploads are limited to 512 MB compressed and 4 GB extracted. Unsafe archive paths and special files are rejected. Before uploading an extracted dataset, you can validate, clean, normalize, and package it with the preparation script from the `backend` directory:

```powershell
python prepare_dataset.py "C:\path\to\extracted-dataset" --format detector --output "C:\path\to\dataset.zip"
```

`--format detector` accepts YOLO `data.yaml`/`data.yml` datasets and Roboflow COCO train/valid annotations and exports normalized YOLO detector data. `--format posture` accepts labeled image folders such as `train/standing/`, `train/lateral_lying/`, `val/standing/`, and `val/lateral_lying/`; Roboflow's `valid/` or `validation/` can be used instead of `val/`. The posture trainer also accepts a `manifest.csv` or `labels.csv` with a posture label in the first column and an image path in the next column; an optional `split` and `source_image` column can specify split and source-frame grouping. The default `--format auto` detects the folder layout. Missing or corrupt images and invalid annotation rows are skipped with warnings; irrecoverably corrupt JSON or inconsistent class/split definitions produce an error instead of a misleading archive. Output is rejected if it exceeds 512 MB compressed or 4 GB uncompressed. Detector bounding-box annotations do not encode posture labels, so the script will not invent posture classes from a COCO or YOLO detection dataset.

Upload the resulting ZIP. To train an object detector, choose **Train detector** for either a YOLO dataset or a Roboflow COCO export. COCO boxes and images are converted to YOLO format in a temporary training workspace; uploaded files are not modified. The backend fine-tunes the current detector for 20 epochs at 640px by default, displays live progress, and atomically activates the new best weights for camera detection. Training runs locally and can take a while. To train posture recognition, use a Roboflow image-classification ZIP with posture-named folders under `train/` and `valid/` (or `val/`/`validation/`) and choose **Train posture**. Roboflow COCO posture exports are also supported: annotated bounding boxes with posture category labels are cropped into classification folders in a temporary workspace. Classes marked `uncertain` or `unknown`, and classes without images in both train and validation, are excluded. The uploaded archive is not modified. A CSV-based dataset can instead include `manifest.csv` or `labels.csv` with the posture label as its first column and the relative image path as its second column. At least two posture classes with images in both training and validation are required. COCO labels such as `Pig 1` and `Pig 2` identify pigs rather than posture and cannot be used to infer posture classes; label posture categories or use the review-crop workflow before training. A dataset cannot be deleted while it is being trained; other uploaded datasets can be removed from the list.

To create a manual posture-review bundle from an uploaded COCO dataset, run:

```powershell
python .\backend\extract_posture_review.py ".\aitrain\datasets\<uploaded-dataset-folder>" --output ".\aitrain\posture-review-crops.zip"
```

The bundle contains crops from `Pig 1`–`Pig 4` annotations, plus `manifest.csv` and review instructions. It deliberately leaves posture labels blank. Launch the visual labeling window from the project root:

```powershell
python .\backend\label_posture_review.py .\aitrain\posture-review-crops.zip
```

The default mutually exclusive classes are `standing`, `sitting`, `sternal_lying` (chest/belly down), and `lateral_lying` (lying on the side). Click a posture button or press its number key (1–4); press **S** to leave an unclear or occluded crop unlabeled, and **Previous** to revisit a crop. Progress is saved in a `.labels.csv` sidecar beside the source ZIP. **Save ZIP and close** exports `posture-review-crops-labeled.zip` without changing the original. Review each crop and assign its actual posture. Upload the labeled bundle and choose **Train posture**. The backend creates train/validation folders automatically, keeping crops from each source frame together to avoid data leakage. Each class needs labeled crops from at least two different source frames, and at least two posture classes are required. Alternatively, put labeled images in matching `train/<posture>/` and `val/<posture>/` class folders; nested dataset folders are supported.

If the dashboard reports `Supabase rejected the configured API key`, copy the current secret key again from **Settings → API Keys** into the root `.env`, then restart both the backend and Vite. Do not send the key in chat. Rotate any secret key that has been shared or exposed.

To remove login-related objects from a database where the earlier schema was already applied, run [`supabase/migrations/002_remove_auth_profiles.sql`](./supabase/migrations/002_remove_auth_profiles.sql) in the Supabase SQL Editor. It drops the app's `public.profiles` table and its Auth-user profile trigger/function; it does not delete Supabase Auth users or disable Supabase Auth for other applications in the project.
