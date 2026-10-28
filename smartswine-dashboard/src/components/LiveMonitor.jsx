import { useEffect, useRef, useState } from 'react';
import { Camera, CheckCircle2, Database, Eye, Play, RefreshCw, Scale, ScanLine, Square, Thermometer } from 'lucide-react';
import { api } from '../api.js';

const FRAME_WIDTH = 960;
const DETECTION_INTERVAL_MS = 5000;
const GO2RTC_CAMERA_ID = 'go2rtc:tapo_c230';

function thermalColor(hue) {
  const segment = hue / 60;
  const x = 1 - Math.abs((segment % 2) - 1);
  const color = segment < 1
    ? [1, x, 0]
    : segment < 2
      ? [x, 1, 0]
      : segment < 3
        ? [0, 1, x]
        : segment < 4
          ? [0, x, 1]
          : segment < 5
            ? [x, 0, 1]
            : [1, 0, x];
  return color.map((channel) => Math.round(channel * 255));
}

function waitForIceGathering(peerConnection) {
  if (peerConnection.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      peerConnection.removeEventListener('icegatheringstatechange', handleStateChange);
      reject(new Error('Timed out preparing the WebRTC camera connection.'));
    }, 10000);
    function handleStateChange() {
      if (peerConnection.iceGatheringState !== 'complete') return;
      window.clearTimeout(timeout);
      peerConnection.removeEventListener('icegatheringstatechange', handleStateChange);
      resolve();
    }
    peerConnection.addEventListener('icegatheringstatechange', handleStateChange);
    handleStateChange();
  });
}

function createRemoteStreamWaiter(peerConnection) {
  let resolveStream;
  let timeout;
  const stream = new Promise((resolve) => {
    resolveStream = resolve;
  });
  function handleTrack(event) {
    peerConnection.removeEventListener('track', handleTrack);
    resolveStream(event.streams[0] || new MediaStream([event.track]));
  }
  peerConnection.addEventListener('track', handleTrack);

  return {
    wait: () => Promise.race([
      stream,
      new Promise((_, reject) => {
        timeout = window.setTimeout(() => {
          peerConnection.removeEventListener('track', handleTrack);
          reject(new Error('go2rtc connected but did not provide a video track.'));
        }, 15000);
      }),
    ]).finally(() => window.clearTimeout(timeout)),
    cancel: () => {
      window.clearTimeout(timeout);
      peerConnection.removeEventListener('track', handleTrack);
    },
  };
}

export default function LiveMonitor({
  pigs,
  observations,
  onRefresh,
  loading,
  offline = false,
  detectorAvailable = false,
  detectorMessage = '',
  postureStatus,
  onRecordDetection,
}) {
  const videoRef = useRef(null);
  const thermalCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [stream, setStream] = useState(null);
  const [activeSource, setActiveSource] = useState(null);
  const [cameraState, setCameraState] = useState('stopped');
  const [cameraError, setCameraError] = useState('');
  const [cameraFrameWarning, setCameraFrameWarning] = useState('');
  const [detectionError, setDetectionError] = useState('');
  const [videoAspectRatio, setVideoAspectRatio] = useState('16 / 9');
  const [detectionEnabled, setDetectionEnabled] = useState(true);
  const [detections, setDetections] = useState([]);
  const [inferenceMs, setInferenceMs] = useState(null);
  const [saving, setSaving] = useState(false);
  const [savedMessage, setSavedMessage] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [thermalStatus, setThermalStatus] = useState({
    available: false,
    message: 'Checking for the MLX90640 sensor…',
  });
  const [thermalViewEnabled, setThermalViewEnabled] = useState(false);
  const [activePostureStatus, setActivePostureStatus] = useState(postureStatus);

  useEffect(() => {
    setActivePostureStatus(postureStatus);
  }, [postureStatus]);

  useEffect(() => {
    let cancelled = false;
    async function refreshThermalStatus() {
      try {
        const status = await api.camera.thermalStatus();
        if (!cancelled) setThermalStatus(status);
      } catch (error) {
        if (!cancelled) {
          setThermalStatus({
            available: false,
            message: error.message || 'Could not check the thermal sensor.',
          });
        }
      }
    }

    void refreshThermalStatus();
    const timer = window.setInterval(() => void refreshThermalStatus(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const canvas = thermalCanvasRef.current;
    const pixels = thermalStatus.pixels;
    if (!canvas || !thermalStatus.available || !Array.isArray(pixels) || pixels.length !== 32 * 24) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const minimum = Math.min(...pixels);
    const maximum = Math.max(...pixels);
    const range = Math.max(maximum - minimum, 0.1);
    const image = context.createImageData(32, 24);
    for (let index = 0; index < pixels.length; index += 1) {
      const value = Math.max(0, Math.min(1, (pixels[index] - minimum) / range));
      const hue = (1 - value) * 240;
      const [red, green, blue] = thermalColor(hue);
      const offset = index * 4;
      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = 255;
    }

    const source = document.createElement('canvas');
    source.width = 32;
    source.height = 24;
    source.getContext('2d')?.putImageData(image, 0, 0);
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
  }, [thermalStatus, thermalViewEnabled]);

  useEffect(() => {
    void refreshCameras();
    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices?.addEventListener) return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      peerConnectionRef.current?.close();
    };
    const handleDeviceChange = () => void refreshCameras();
    mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => {
      mediaDevices.removeEventListener('devicechange', handleDeviceChange);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      peerConnectionRef.current?.close();
    };
  }, []);

  useEffect(() => () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  }, [imageUrl]);

  useEffect(() => {
    if (!videoRef.current || !stream) return;
    videoRef.current.srcObject = stream;
    videoRef.current.play().catch((error) => {
      setCameraError(`Could not start camera preview: ${error.message}`);
    });
  }, [stream, thermalViewEnabled]);

  useEffect(() => {
    if (!stream || !detectionEnabled || !detectorAvailable) return undefined;
    let cancelled = false;
    let inFlight = false;
    const canvas = document.createElement('canvas');
    const diagnosticCanvas = document.createElement('canvas');
    diagnosticCanvas.width = 24;
    diagnosticCanvas.height = 24;
    const diagnosticContext = diagnosticCanvas.getContext('2d', { willReadFrequently: true });

    async function analyzeFrame() {
      const video = videoRef.current;
      if (cancelled || inFlight || !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      inFlight = true;
      try {
        const scale = Math.min(1, FRAME_WIDTH / video.videoWidth);
        canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Could not prepare the camera frame for detection.');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (!diagnosticContext) throw new Error('Could not inspect the camera frame.');
        diagnosticContext.drawImage(video, 0, 0, diagnosticCanvas.width, diagnosticCanvas.height);
        const pixels = diagnosticContext.getImageData(
          0,
          0,
          diagnosticCanvas.width,
          diagnosticCanvas.height,
        ).data;
        let brightestPixel = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          brightestPixel = Math.max(
            brightestPixel,
            pixels[index],
            pixels[index + 1],
            pixels[index + 2],
          );
        }
        if (brightestPixel <= 8) {
          if (!cancelled) {
            setCameraFrameWarning(
              'The camera is sending a black image. Check its privacy shutter or camera privacy mode, and close other apps using it.',
            );
            setDetections([]);
            setSavedMessage('');
          }
          return;
        }
        setCameraFrameWarning('');
        const frame = await new Promise((resolve, reject) => {
          canvas.toBlob((blob) => {
            if (blob) resolve(blob);
            else reject(new Error('Could not encode the camera frame.'));
          }, 'image/jpeg', 0.78);
        });
        const result = await api.camera.detect(frame);
        if (cancelled) return;
        setDetections(result.detections);
        setActivePostureStatus(result.posture_status || postureStatus);
        setInferenceMs(result.inference_ms);
        setDetectionError('');
        setSavedMessage('');
        if (result.detections.length) {
          setSaving(true);
          let savedCount = 0;
          try {
            for (const detection of result.detections) {
              if (cancelled) return;
              await onRecordDetection(detection);
              savedCount += 1;
            }
            if (!cancelled) {
              setSavedMessage(
                `Saved ${savedCount} detection${savedCount === 1 ? '' : 's'} locally; syncing automatically when connected.`,
              );
            }
          } catch (error) {
            if (!cancelled) {
              setDetectionError(
                `Automatic save stopped after ${savedCount} detection${savedCount === 1 ? '' : 's'}: ${error.message}`,
              );
            }
          } finally {
            if (!cancelled) setSaving(false);
          }
        }
      } catch (error) {
        if (!cancelled) setDetectionError(error.message || 'Pig detection failed.');
      } finally {
        inFlight = false;
      }
    }

    void analyzeFrame();
    const timer = window.setInterval(() => void analyzeFrame(), DETECTION_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [stream, detectionEnabled, detectorAvailable, selectedDeviceId, postureStatus, offline, onRecordDetection]);

  async function refreshCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setCameraError('This browser does not support camera selection.');
      return;
    }
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter((device) => device.kind === 'videoinput');
      setCameraDevices(devices);
      setSelectedDeviceId((current) => current || devices[0]?.deviceId || '');
    } catch (error) {
      setCameraError(`Could not list cameras: ${error.message}`);
    }
  }

  function stopCamera() {
    peerConnectionRef.current?.close();
    peerConnectionRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStream(null);
    setActiveSource(null);
    setCameraState('stopped');
    setCameraFrameWarning('');
    setDetections([]);
    setInferenceMs(null);
    setDetectionError('');
  }

  async function startCamera(deviceId = selectedDeviceId) {
    setCameraError('');
    setCameraFrameWarning('');
    setDetectionError('');
    setCameraState('connecting');
    peerConnectionRef.current?.close();
    peerConnectionRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStream(null);
    setActiveSource(null);
    let remoteStreamWaiter;
    try {
      if (deviceId === GO2RTC_CAMERA_ID) {
        if (!window.RTCPeerConnection) {
          throw new Error('This browser does not support WebRTC camera streams.');
        }
        const peerConnection = new RTCPeerConnection({ iceServers: [] });
        peerConnectionRef.current = peerConnection;
        peerConnection.addTransceiver('video', { direction: 'recvonly' });
        remoteStreamWaiter = createRemoteStreamWaiter(peerConnection);
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        await waitForIceGathering(peerConnection);
        if (!peerConnection.localDescription?.sdp) {
          throw new Error('The browser could not create a WebRTC camera offer.');
        }
        const answer = await api.camera.tapoWebRTCOffer(peerConnection.localDescription.sdp);
        await peerConnection.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
        const remoteStream = await remoteStreamWaiter.wait();
        streamRef.current = remoteStream;
        setStream(remoteStream);
        setActiveSource('go2rtc');
      } else {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Camera access requires a supported browser on localhost or HTTPS.');
        }
        const videoConstraints = deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } };
        const nextStream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false,
        });
        streamRef.current = nextStream;
        setStream(nextStream);
        setActiveSource('browser');
        await refreshCameras();
      }
      setCameraState('connected');
      setDetections([]);
      setSavedMessage('');
    } catch (error) {
      remoteStreamWaiter?.cancel();
      peerConnectionRef.current?.close();
      peerConnectionRef.current = null;
      streamRef.current = null;
      setStream(null);
      setActiveSource(null);
      setCameraState('stopped');
      const message = error.name === 'NotAllowedError'
        ? 'Allow camera access in your browser, then try again.'
        : error.name === 'NotFoundError'
          ? 'No camera was found. Connect a camera and try again.'
          : error.name === 'NotReadableError'
            ? 'The camera is already in use by another app.'
            : error.message;
      setCameraError(
        deviceId === GO2RTC_CAMERA_ID
          ? `${message} Confirm go2rtc is running and the Tapo RTSP stream is reachable.`
          : message,
      );
    }
  }

  function handleCameraChange(event) {
    const deviceId = event.target.value;
    setSelectedDeviceId(deviceId);
    if (!deviceId) {
      stopCamera();
      return;
    }
    if (deviceId === GO2RTC_CAMERA_ID || cameraState === 'connected' || streamRef.current) {
      void startCamera(deviceId);
    }
  }

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setCameraError('Choose an image file.');
      return;
    }
    setCameraError('');
    setImageUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(file);
    });
    event.target.value = '';
  }

  const previewActive = cameraState === 'connected' && Boolean(stream);
  const go2rtcPreviewActive = cameraState === 'connected' && activeSource === 'go2rtc';
  const cameraLabel = go2rtcPreviewActive
    ? 'TAPO C230 CONNECTED'
    : cameraState === 'connected'
      ? 'CAMERA CONNECTED'
    : cameraState === 'connecting'
      ? 'CONNECTING'
      : 'CAMERA OFF';
  const thermalPixels = thermalStatus.pixels;
  const thermalMinimum = thermalStatus.available && Array.isArray(thermalPixels)
    ? Math.min(...thermalPixels).toFixed(1)
    : null;
  const thermalMaximum = thermalStatus.available && Array.isArray(thermalPixels)
    ? Math.max(...thermalPixels).toFixed(1)
    : null;

  return (
    <div className="page-grid live-grid">
      <section className="panel camera-panel">
        <div className="panel-heading">
          <div><Camera className="accent-icon" /><div><h2>Live pen camera</h2><p>{go2rtcPreviewActive ? 'Tapo C230 stream via go2rtc with pig detection enabled.' : previewActive ? 'Live video from the selected camera.' : 'Choose a camera and allow browser access to start live monitoring.'}</p></div></div>
          <span className={`pill ${thermalViewEnabled && thermalStatus.available ? 'pill-green' : cameraState === 'connected' ? 'pill-green' : cameraState === 'connecting' ? 'pill-amber' : 'pill-red'}`}>{thermalViewEnabled ? (thermalStatus.available ? 'MLX90640 LIVE' : 'MLX90640 WAITING') : cameraLabel}</span>
        </div>
        <div
          className="camera-view"
          style={{ aspectRatio: thermalViewEnabled ? '4 / 3' : previewActive ? videoAspectRatio : '16 / 9' }}
        >
          {thermalViewEnabled
            ? thermalStatus.available && Array.isArray(thermalPixels)
              ? <canvas ref={thermalCanvasRef} className="thermal-camera-canvas" width="320" height="240" aria-label="Live MLX90640 thermal image" />
              : <div className="camera-placeholder"><Thermometer size={42} /><strong>MLX90640 frame not available</strong><span>{thermalStatus.message || 'Waiting for a recent thermal frame from the ESP32-S3.'}</span></div>
            : previewActive
              ? <video ref={videoRef} className="camera-video" autoPlay muted playsInline aria-label="Live selected camera" onLoadedMetadata={(event) => setVideoAspectRatio(`${event.currentTarget.videoWidth} / ${event.currentTarget.videoHeight}`)} />
              : imageUrl
                ? <img className="uploaded-image" src={imageUrl} alt="Uploaded image preview" />
                : <div className="camera-placeholder"><Camera size={42} /><strong>{cameraState === 'connecting' ? 'Connecting to camera…' : 'Camera is off'}</strong><span>{cameraError || (cameraDevices.length ? 'Select a camera below and start the live stream.' : 'Allow camera permission to list available cameras.')}</span></div>}
          {previewActive && !thermalViewEnabled && detections.map((detection, index) => (
            <div
              className="detection-box"
              key={`${detection.class_name}-${index}`}
              style={{ left: `${detection.box.x}%`, top: `${detection.box.y}%`, width: `${detection.box.w}%`, height: `${detection.box.h}%` }}
            >
              <span className="detection-tag"><Eye size={13} />{detection.class_name} · {Math.round(detection.confidence * 100)}%{detection.weight_estimate ? ` · ~${detection.weight_estimate.weight_kg} kg` : ''}{detection.surface_temperature_c != null ? ` · ${detection.surface_temperature_c.toFixed(1)}°C surface` : ''}</span>
              {detection.posture && (
                <span className="posture-tag">
                  Posture: {detection.posture.label} · {Math.round(detection.posture.confidence * 100)}%
                </span>
              )}
            </div>
          ))}
          {previewActive && !thermalViewEnabled && !detections.length && detectorAvailable && detectionEnabled && (
            <span className="camera-overlay-status"><ScanLine size={14} />Searching for pigs…</span>
          )}
          {previewActive && !thermalViewEnabled && cameraFrameWarning && (
            <span className="camera-frame-warning">{cameraFrameWarning}</span>
          )}
          {thermalViewEnabled && thermalStatus.available && thermalMinimum != null && thermalMaximum != null && (
            <div className="thermal-image-legend">
              <span>Cool {thermalMinimum}°C</span><span className="thermal-gradient" aria-hidden="true" /><span>Warm {thermalMaximum}°C</span>
              {thermalStatus.ambient_c != null && <span>Ambient {thermalStatus.ambient_c.toFixed(1)}°C</span>}
            </div>
          )}
        </div>
        <section className="weight-estimation-settings">
          <div className="weight-estimation-heading">
            <div><strong>Approximate camera weight</strong><span>Fixed overhead camera height: 2 m</span></div>
            <span className="pill pill-green">Automatic</span>
          </div>
          <p>Weight estimation</p>
          {detections.some((detection) => detection.weight_estimate) && (
            <div className="estimated-weight-list">
              {detections.filter((detection) => detection.weight_estimate).map((detection, index) => (
                <span key={`${detection.class_name}-weight-${index}`}>
                  {detection.class_name}: ~{detection.weight_estimate.weight_kg} kg
                </span>
              ))}
            </div>
          )}
        </section>
        <section className="posture-identifier">
          <div className="weight-estimation-heading">
            <div><strong>Posture identifier</strong><span>Trained image classifier on detected pig crops</span></div>
            <span className={`pill ${activePostureStatus?.available ? 'pill-green' : 'pill-amber'}`}>
              {activePostureStatus?.available ? 'Active' : 'Model required'}
            </span>
          </div>
          <p>{activePostureStatus?.available
            ? `Model labels: ${activePostureStatus.classes.join(', ')}. Results are visual observations, not veterinary diagnoses.`
            : activePostureStatus?.message || 'Set PIG_POSTURE_MODEL to trained YOLO image-classification weights, then restart the backend.'}</p>
          {detections.some((detection) => detection.posture) && (
            <div className="estimated-weight-list">
              {detections.filter((detection) => detection.posture).map((detection, index) => (
                <span key={`${detection.class_name}-posture-${index}`}>
                  {detection.class_name}: {detection.posture.label} ({Math.round(detection.posture.confidence * 100)}%)
                </span>
              ))}
            </div>
          )}
          <p></p>
        </section>
        <div className="camera-controls camera-setup">
          <button
            className={`button ${thermalViewEnabled ? 'button-primary' : 'button-secondary'}`}
            onClick={() => setThermalViewEnabled((enabled) => !enabled)}
            aria-pressed={thermalViewEnabled}
          >
            <Thermometer size={15} />{thermalViewEnabled ? 'Show RGB camera' : 'MLX90640 thermal view'}
          </button>
          <label className="camera-select-label" htmlFor="camera-device">Camera source
            <select id="camera-device" value={selectedDeviceId} onChange={handleCameraChange} disabled={cameraState === 'connecting'}>
              {!cameraDevices.length && <option value="">No browser camera found</option>}
              {cameraDevices.map((device, index) => (
                <option value={device.deviceId} key={device.deviceId}>{device.label || `Camera ${index + 1}`}</option>
              ))}
              <option value={GO2RTC_CAMERA_ID}>Tapo C230 (go2rtc)</option>
            </select>
          </label>
          {cameraState === 'connected'
            ? <button className="button button-secondary" onClick={stopCamera}><Square size={15} />Stop camera</button>
            : <button className="button button-primary" onClick={() => void startCamera()} disabled={cameraState === 'connecting'}><Play size={15} />{cameraState === 'connecting' ? 'Connecting…' : 'Start camera'}</button>}
          <label className="camera-detection-toggle">
            <input type="checkbox" checked={detectionEnabled} onChange={(event) => setDetectionEnabled(event.target.checked)} disabled={!detectorAvailable} />
            Detect pigs
          </label>
          <label className="button button-secondary upload-button">Preview image<input type="file" accept="image/*" onChange={handleUpload} /></label>
        </div>
        <div className="camera-model-status">
          <span className={`status-dot ${detectorAvailable ? '' : 'status-offline'}`} />
          {detectorAvailable
            ? `Pig detector ready${inferenceMs == null ? '' : ` · ${inferenceMs} ms`}`
            : detectorMessage || 'Pig detector is not available.'}
          {detectorAvailable && <span className="automatic-save-status">{saving ? 'Saving observations…' : 'Auto-saving every 5 seconds'}</span>}
        </div>
        <div className="camera-model-status thermal-status">
          <span className={`status-dot ${thermalStatus.available ? '' : 'status-offline'}`} />
          {thermalStatus.available
            ? `MLX90640 surface scan ready${thermalStatus.age_seconds == null ? '' : ` · ${thermalStatus.age_seconds}s old`}`
            : thermalStatus.message}
          <span className="automatic-save-status">Apparent surface temperature only—not core/body temperature.</span>
        </div>
        {savedMessage && <p className="detection-success"><CheckCircle2 size={15} />{savedMessage}</p>}
        {(cameraError || detectionError) && (
          <p className="inline-error">{detectionError || cameraError}</p>
        )}
        <div className="camera-controls">
          <button className="button button-secondary" onClick={onRefresh} disabled={loading}><RefreshCw size={16} />{loading ? 'Refreshing…' : 'Refresh observations and sync'}</button>
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading"><div><span className="activity-icon"><DatabaseIcon /></span><div><h2>Recorded observations</h2><p>{observations.length} locally saved records</p></div></div></div>
        <div className="observation-list">
          {observations.map((item) => {
            const pig = pigs.find((entry) => entry.id === item.pigId);
            return <article className="observation-card" key={item.id}>
              <div className="row-between"><strong className="identity">{item.aiIdentity || 'Unassigned'}</strong><time>{item.timestamp}</time></div>
              <div className="observation-details"><span>Pen <b>{pig?.penTag ?? item.penTag ?? '—'}</b></span><span>{item.behavior || '—'}</span></div>
              {item.surfaceTemperatureC != null && (
                <div className="observation-footer thermal-observation">
                  Apparent surface temperature: {item.surfaceTemperatureC.toFixed(1)}°C
                </div>
              )}
              <div className="row-between observation-footer"><span><Scale size={14} />{item.weight == null ? '—' : `${Number(item.weight).toFixed(1)} kg`}</span><span>{item.confidence == null ? '—' : `${Math.round(item.confidence * 100)}% confidence`}</span></div>
            </article>;
          })}
          {!observations.length && <p className="empty-state">{offline ? 'No observations are saved on this device yet.' : 'No observations have been recorded.'}</p>}
        </div>
      </section>
    </div>
  );
}

function DatabaseIcon() {
  return <RefreshCw size={17} />;
}
