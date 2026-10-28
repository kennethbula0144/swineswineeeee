import { useEffect, useRef, useState } from 'react';
import { Archive, Cpu, Database, Play, RefreshCw, Trash2, Upload } from 'lucide-react';
import { api } from '../api.js';

export default function Settings({ databaseStatus, onRefresh, loading, onDetectorUpdated }) {
  const fileInputRef = useRef(null);
  const [datasets, setDatasets] = useState([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [datasetError, setDatasetError] = useState('');
  const [datasetMessage, setDatasetMessage] = useState('');
  const [training, setTraining] = useState(null);
  const [postureTraining, setPostureTraining] = useState(null);
  const [trainingError, setTrainingError] = useState('');
  const previousTrainingStatus = useRef('');
  const previousPostureTrainingStatus = useRef('');
  const databaseConnected = databaseStatus === 'connected';
  const statusLabel = databaseConnected ? 'Connected' : databaseStatus === 'connecting' ? 'Connecting' : 'Unavailable';

  async function refreshDatasets() {
    setDatasetsLoading(true);
    setDatasetError('');
    try {
      setDatasets(await api.datasets.list());
    } catch (error) {
      setDatasetError(error.message || 'Could not load uploaded datasets.');
    } finally {
      setDatasetsLoading(false);
    }
  }

  useEffect(() => {
    void refreshDatasets();
    let checking = false;
    async function refreshTrainingStatus() {
      if (checking) return;
      checking = true;
      try {
        const [detectorResult, postureResult] = await Promise.allSettled([
          api.training.status(),
          api.training.postureStatus(),
        ]);
        if (detectorResult.status === 'fulfilled') {
          const next = detectorResult.value;
          setTraining(next);
          if (previousTrainingStatus.current !== 'completed' && next.status === 'completed') {
            onDetectorUpdated?.();
            setDatasetMessage('Training finished. The new model is now active for object detection.');
          }
          previousTrainingStatus.current = next.status;
        } else {
          setTrainingError(detectorResult.reason.message || 'Could not check detector training status.');
        }
        if (postureResult.status === 'fulfilled') {
          const nextPosture = postureResult.value;
          setPostureTraining(nextPosture);
          if (previousPostureTrainingStatus.current !== 'completed' && nextPosture.status === 'completed') {
            onDetectorUpdated?.();
            setDatasetMessage('Posture training finished. The new classifier is now active.');
          }
          previousPostureTrainingStatus.current = nextPosture.status;
          if (detectorResult.status === 'fulfilled') setTrainingError('');
        } else {
          setTrainingError(postureResult.reason.message || 'Could not check posture training status.');
        }
      } catch (error) {
        setTrainingError(error.message || 'Could not check model training status.');
      } finally {
        checking = false;
      }
    }
    void refreshTrainingStatus();
    const timer = window.setInterval(() => void refreshTrainingStatus(), 1500);
    return () => window.clearInterval(timer);
  }, [onDetectorUpdated]);

  async function uploadDataset(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setDatasetError('');
    setDatasetMessage('');
    if (!file.name.toLowerCase().endsWith('.zip')) {
      setDatasetError('Choose a dataset archive with a .zip extension.');
      return;
    }
    if (file.size > 512 * 1024 * 1024) {
      setDatasetError('ZIP files must be 512 MB or smaller.');
      return;
    }

    setUploading(true);
    try {
      const uploaded = await api.datasets.upload(file);
      setDatasets((current) => [uploaded, ...current]);
      setDatasetMessage(`${uploaded.name} uploaded and extracted successfully.`);
    } catch (error) {
      setDatasetError(error.message || 'Dataset upload failed.');
    } finally {
      setUploading(false);
    }
  }

  async function startTraining(dataset) {
    setTrainingError('');
    setDatasetMessage('');
    try {
      const started = await api.datasets.train(dataset.id);
      previousTrainingStatus.current = started.status;
      setTraining(started);
    } catch (error) {
      setTrainingError(error.message || 'Could not start YOLO training.');
    }
  }

  async function startPostureTraining(dataset) {
    setTrainingError('');
    setDatasetMessage('');
    try {
      const started = await api.datasets.trainPosture(dataset.id);
      previousPostureTrainingStatus.current = started.status;
      setPostureTraining(started);
    } catch (error) {
      setTrainingError(error.message || 'Could not start posture training.');
    }
  }

  async function deleteDataset(dataset) {
    if (!window.confirm(`Delete "${dataset.name}" and all extracted files? This cannot be undone.`)) return;
    setDatasetError('');
    setDatasetMessage('');
    try {
      await api.datasets.delete(dataset.id);
      setDatasets((current) => current.filter((entry) => entry.id !== dataset.id));
      setDatasetMessage(`${dataset.name} and its extracted files were deleted.`);
    } catch (error) {
      setDatasetError(error.message || 'Could not delete the dataset.');
    }
  }

  const trainingRunning = training?.status === 'running';
  const postureTrainingRunning = postureTraining?.status === 'running';
  const anyTrainingRunning = trainingRunning || postureTrainingRunning;
  const trainingProgress = trainingRunning && training.epochs > 0
    ? Math.min(100, Math.round((training.epoch / training.epochs) * 100))
    : training?.status === 'completed' ? 100 : 0;

  return <div className="settings-stack">
    <section className="panel settings-panel">
      <h2><Database className="accent-icon" />Supabase connection</h2>
      <div className="setting-row">
        <div><strong>Database status</strong><p>Supabase is accessed through the FastAPI backend.</p></div>
        <span className={`pill ${databaseConnected ? 'pill-green' : databaseStatus === 'connecting' ? 'pill-amber' : 'pill-red'}`}>{statusLabel}</span>
      </div>
      <p className="chart-note">No Supabase sign-in is required when the app and API are running on this machine. The API rejects remote clients, and the Supabase secret stays on the backend.</p>
      <button className="button button-secondary" onClick={() => void onRefresh()} disabled={loading}>{loading ? 'Checking…' : 'Check connection'}</button>
    </section>
    <section className="panel settings-panel">
      <h2><Archive className="accent-icon" />Dataset uploads</h2>
      <p>Upload a ZIP exported from Roboflow or another dataset tool. Roboflow COCO exports with train/valid folders and <code>_annotations.coco.json</code> files are converted automatically; YOLO exports with one data.yaml are also supported for pig detection. To train posture recognition, upload an image-classification ZIP with posture-named image folders under <code>train/standing/</code>, <code>train/lateral_lying/</code>, <code>valid/standing/</code>, and <code>valid/lateral_lying/</code> (the validation folder may also be named <code>val/</code> or <code>validation/</code>; nested ZIP wrapper folders are supported). Roboflow COCO posture annotations are supported too: the trainer crops each labeled box into a class folder in a temporary workspace. Classes named <code>uncertain</code> or <code>unknown</code>, and classes missing from either split, are excluded. COCO labels such as <code>Pig 1</code> identify pigs rather than their posture and cannot be used to infer posture classes. A CSV alternative is <code>manifest.csv</code> or <code>labels.csv</code> with the posture <code>label</code> in the first column and the image path in the second. The trainer normalizes Roboflow's <code>valid/</code> split to <code>val/</code> without changing the uploaded files. For the posture-review ZIP, run <code>python .\backend\label_posture_review.py .\aitrain\posture-review-crops.zip</code> from the project root, label each crop in the window, then upload the generated <code>posture-review-crops-labeled.zip</code>. Label the pig's primary position as <code>standing</code>, <code>sitting</code>, <code>sternal_lying</code> (chest/belly down), or <code>lateral_lying</code> (lying on its side). These classes are mutually exclusive; skip unclear or occluded crops. Labels and progress are saved as you go. The backend splits crops from the same source frame together. At least two posture classes and images from multiple source frames per class are required. ZIP files may be up to 512 MB compressed and 4 GB extracted.</p>
      <div className="dataset-upload-actions">
        <input ref={fileInputRef} className="dataset-file-input" type="file" accept=".zip,application/zip" onChange={(event) => void uploadDataset(event)} />
        <button className="button button-primary" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
          <Upload size={15} />{uploading ? 'Uploading and extracting…' : 'Choose dataset ZIP'}
        </button>
        <button className="button button-secondary" onClick={() => void refreshDatasets()} disabled={datasetsLoading || uploading} aria-label="Refresh uploaded datasets">
          <RefreshCw size={15} />{datasetsLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      {datasetError && <p className="dataset-error" role="alert">{datasetError}</p>}
      {datasetMessage && <p className="success-message" role="status">{datasetMessage}</p>}
      {trainingError && <p className="dataset-error" role="alert">{trainingError}</p>}
      {training && training.status !== 'idle' && (
        <div className="training-status" role="status">
          <div className="training-status-heading"><strong><Cpu size={15} />{trainingRunning ? 'Training pig detector' : training.status === 'completed' ? 'Detector training complete' : 'Detector training failed'}</strong><span>{trainingRunning ? `${training.epoch} / ${training.epochs} epochs` : ''}</span></div>
          {trainingRunning && <div className="progress-track"><span style={{ width: `${trainingProgress}%` }} /></div>}
          <p className={training.status === 'failed' ? 'dataset-error' : ''}>{training.status === 'failed' ? training.error || training.message : training.message}</p>
        </div>
      )}
      {postureTraining && postureTraining.status !== 'idle' && (
        <div className="training-status" role="status">
          <div className="training-status-heading"><strong><Cpu size={15} />{postureTrainingRunning ? 'Training posture classifier' : postureTraining.status === 'completed' ? 'Posture training complete' : 'Posture training failed'}</strong><span>{postureTrainingRunning ? `${postureTraining.epoch} / ${postureTraining.epochs} epochs` : ''}</span></div>
          {postureTrainingRunning && <div className="progress-track"><span style={{ width: `${Math.min(100, Math.round((postureTraining.epoch / postureTraining.epochs) * 100))}%` }} /></div>}
          <p className={postureTraining.status === 'failed' ? 'dataset-error' : ''}>{postureTraining.status === 'failed' ? postureTraining.error || postureTraining.message : postureTraining.message}</p>
        </div>
      )}
      <div className="dataset-list" aria-live="polite">
        {datasetsLoading
          ? <p className="chart-note">Loading uploaded datasets…</p>
          : datasets.length
            ? datasets.map((dataset) => (
              <div className="setting-row dataset-row" key={dataset.id}>
                <div><strong>{dataset.name}</strong><p>{dataset.file_count.toLocaleString()} files · {(dataset.size_bytes / (1024 * 1024)).toFixed(1)} MB extracted · {new Date(dataset.uploaded_at).toLocaleString()}</p></div>
                <div className="dataset-actions">
                  {training?.dataset_id === dataset.id && training.status === 'completed'
                    ? <span className="pill pill-green">Active detector</span>
                    : <button className="button button-secondary" onClick={() => void startTraining(dataset)} disabled={anyTrainingRunning || uploading || datasetsLoading} title="Train and activate a detector using this dataset">
                      <Play size={14} />{trainingRunning && training.dataset_id === dataset.id ? 'Training…' : 'Train detector'}
                    </button>}
                  {postureTraining?.dataset_id === dataset.id && postureTraining.status === 'completed'
                    ? <span className="pill pill-green">Active posture model</span>
                    : <button className="button button-secondary" onClick={() => void startPostureTraining(dataset)} disabled={anyTrainingRunning || uploading || datasetsLoading} title="Train posture recognition using labeled image folders">
                      <Play size={14} />{postureTrainingRunning && postureTraining.dataset_id === dataset.id ? 'Training…' : 'Train posture'}
                    </button>}
                  <button className="button button-danger" onClick={() => void deleteDataset(dataset)} disabled={anyTrainingRunning && (training?.dataset_id === dataset.id || postureTraining?.dataset_id === dataset.id)} aria-label={`Delete ${dataset.name}`} title={anyTrainingRunning && (training?.dataset_id === dataset.id || postureTraining?.dataset_id === dataset.id) ? 'Cannot delete while training is running' : 'Delete uploaded dataset'}>
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))
            : <p className="chart-note">No datasets uploaded yet.</p>}
      </div>
    </section>
    <section className="panel settings-panel">
      <h2>Pig detection</h2>
      <p>The live monitor uses the local YOLO pig detector. Choose an available camera, start the stream, and enable detection. Save detections from the live monitor to add them to Supabase observations.</p>
    </section>
  </div>;
}
