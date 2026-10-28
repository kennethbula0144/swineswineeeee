import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { AlertCircle, Cpu, Database, Scale } from 'lucide-react';
import { api } from './api.js';
import { offlineStore } from './offlineStore.js';
import Sidebar, { MobileMenu, pages } from './components/Sidebar.jsx';
import LiveMonitor from './components/LiveMonitor.jsx';
import PigRegistry from './components/PigRegistry.jsx';
import FeedExpenses from './components/FeedExpenses.jsx';
import HealthAlerts from './components/HealthAlerts.jsx';
import Settings from './components/Settings.jsx';

const Analytics = lazy(() => import('./components/Analytics.jsx'));
const SPLASH_DURATION_MS = 2200;
const RECORD_KEYS = ['pigs', 'observations', 'alerts', 'feedLogs'];

function sortPigs(records) {
  return [...records].sort((left, right) => Number(left.penTag) - Number(right.penTag));
}

function mergePending(remoteRecords, pending, entity) {
  const records = new Map(remoteRecords.map((record) => [record.id, record]));
  for (const operation of pending) {
    if (operation.entity !== entity) continue;
    if (operation.type === 'delete') {
      records.delete(operation.id);
    } else if (operation.type === 'create') {
      records.set(operation.id, operation.payload);
    } else if (operation.type === 'updateStatus') {
      const current = records.get(operation.id);
      if (current) records.set(operation.id, { ...current, status: operation.payload.status });
    }
  }
  return [...records.values()];
}

function mapLocalDetection(detection, pigs) {
  const now = new Date();
  const identity = detection.identity || detection.class_name || detection.className || 'Unassigned';
  const identityKey = identity.replace(/[^a-z0-9]/gi, '').toLowerCase();
  const matches = pigs.filter((pig) => (
    pig.aiIdentity?.replace(/[^a-z0-9]/gi, '').toLowerCase() === identityKey
  ));
  const pig = matches.length === 1 ? matches[0] : null;
  const weight = detection.weight_estimate?.weight_kg;
  return {
    id: detection.id,
    pigId: pig?.id ?? null,
    timestamp: now.toLocaleString(),
    createdAt: now.toISOString(),
    aiIdentity: identity,
    penTag: pig?.penTag ?? '—',
    weight: weight == null ? null : Number(weight),
    surfaceTemperatureC: detection.surface_temperature_c == null
      ? null
      : Number(detection.surface_temperature_c),
    behavior: detection.posture
      ? `Posture: ${detection.posture.label}`
      : detection.weight_estimate
        ? 'Camera weight estimate'
        : 'Unknown',
    confidence: detection.confidence == null ? null : Number(detection.confidence),
    box: detection.box,
  };
}

export default function App() {
  const [splashVisible, setSplashVisible] = useState(true);
  const [activePage, setActivePage] = useState('live');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [pigs, setPigs] = useState([]);
  const [observations, setObservations] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [feedLogs, setFeedLogs] = useState([]);
  const [databaseStatus, setDatabaseStatus] = useState('connecting');
  const [databaseError, setDatabaseError] = useState('');
  const [syncError, setSyncError] = useState('');
  const [pendingSyncCount, setPendingSyncCount] = useState(0);
  const [dataLoading, setDataLoading] = useState(false);
  const [detectorStatus, setDetectorStatus] = useState({ available: false, message: 'Checking pig detector…' });
  const syncLock = useRef(false);
  const page = pages.find((item) => item.id === activePage);
  const databaseConnected = databaseStatus === 'connected';

  const applySnapshot = useCallback((snapshot) => {
    setPigs(snapshot.pigs);
    setObservations(snapshot.observations);
    setAlerts(snapshot.alerts);
    setFeedLogs(snapshot.feedLogs);
    setPendingSyncCount(snapshot.pending.length);
  }, []);

  const refreshData = useCallback(async () => {
    setDataLoading(true);
    setDatabaseError('');
    try {
      const cached = await offlineStore.readSnapshot();
      applySnapshot(cached);

      await api.health();
      const [pigRecords, observationRecords, alertRecords, feedRecords] = await Promise.all([
        api.pigs.list(),
        api.observations.list(),
        api.alerts.list(),
        api.feedLogs.list(),
      ]);
      const latest = await offlineStore.readSnapshot();
      const remote = {
        pigs: sortPigs(mergePending(pigRecords, latest.pending, 'pigs')),
        observations: mergePending(observationRecords, latest.pending, 'observations')
          .sort((left, right) => (right.createdAt || '').localeCompare(left.createdAt || '')),
        alerts: mergePending(alertRecords, latest.pending, 'alerts'),
        feedLogs: mergePending(feedRecords, latest.pending, 'feedLogs'),
      };
      await Promise.all(RECORD_KEYS.map((key) => offlineStore.replaceRecords(key, remote[key])));
      const snapshot = await offlineStore.readSnapshot();
      applySnapshot(snapshot);
      setDatabaseStatus('connected');
      setDatabaseError('');
      return true;
    } catch (error) {
      setDatabaseStatus('disconnected');
      setDatabaseError(error.message || 'Could not load farm data.');
      return false;
    } finally {
      setDataLoading(false);
    }
  }, [applySnapshot]);

  const syncPendingChanges = useCallback(async () => {
    if (syncLock.current || navigator.onLine === false) return;
    syncLock.current = true;
    try {
      let snapshot = await offlineStore.readSnapshot();
      setPendingSyncCount(snapshot.pending.length);
      await api.health();
      setDatabaseStatus('connected');
      setDatabaseError('');
      if (!snapshot.pending.length) {
        setSyncError('');
        return;
      }

      setSyncError('');
      while (snapshot.pending.length) {
        for (const operation of snapshot.pending) {
          try {
            if (operation.entity === 'pigs' && operation.type === 'create') {
              await api.pigs.create(operation.payload);
            } else if (operation.entity === 'pigs' && operation.type === 'delete') {
              await api.pigs.delete(operation.id);
            } else if (operation.entity === 'feedLogs' && operation.type === 'create') {
              await api.feedLogs.create(operation.payload);
            } else if (operation.entity === 'feedLogs' && operation.type === 'delete') {
              await api.feedLogs.delete(operation.id);
            } else if (operation.entity === 'observations' && operation.type === 'create') {
              await api.observations.create(operation.payload);
            } else if (operation.entity === 'alerts' && operation.type === 'updateStatus') {
              await api.alerts.updateStatus(operation.id, operation.payload.status);
            } else {
              throw new Error(`Unsupported pending sync operation: ${operation.entity}.${operation.type}`);
            }
            await offlineStore.removePending(operation.opId);
            setPendingSyncCount((count) => Math.max(0, count - 1));
          } catch (error) {
            if (operation.type === 'delete' && error.status === 404) {
              await offlineStore.removePending(operation.opId);
              setPendingSyncCount((count) => Math.max(0, count - 1));
              continue;
            }
            throw error;
          }
        }
        snapshot = await offlineStore.readSnapshot();
        setPendingSyncCount(snapshot.pending.length);
      }
      setSyncError('');
      await refreshData();
    } catch (error) {
      setSyncError(error.message || 'Could not sync local farm changes.');
      if (navigator.onLine === false || !error.status) setDatabaseStatus('disconnected');
      const count = await offlineStore.pendingCount();
      setPendingSyncCount(count);
    } finally {
      syncLock.current = false;
    }
  }, [refreshData]);

  const refreshAndSync = useCallback(async () => {
    const available = await refreshData();
    if (available) await syncPendingChanges();
    return available;
  }, [refreshData, syncPendingChanges]);

  useEffect(() => {
    const timer = window.setTimeout(() => setSplashVisible(false), SPLASH_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (splashVisible) return undefined;
    void refreshAndSync();
    api.camera.status()
      .then(setDetectorStatus)
      .catch((error) => setDetectorStatus({ available: false, message: error.message }));
    const handleOnline = () => void refreshAndSync();
    const retryTimer = window.setInterval(() => void syncPendingChanges(), 30000);
    window.addEventListener('online', handleOnline);
    return () => {
      window.clearInterval(retryTimer);
      window.removeEventListener('online', handleOnline);
    };
  }, [splashVisible, refreshAndSync, syncPendingChanges]);

  async function createPig(pig) {
    const created = {
      ...pig,
      id: crypto.randomUUID(),
      aiIdentity: pig.aiIdentity || 'Unassigned',
    };
    const operation = offlineStore.createOperation('create', 'pigs', created.id, created);
    await offlineStore.saveAndQueue('pigs', created, operation);
    setPigs((current) => sortPigs([...current, created]));
    setPendingSyncCount((count) => count + 1);
    void syncPendingChanges();
  }

  async function deletePig(id) {
    const operation = offlineStore.createOperation('delete', 'pigs', id);
    await offlineStore.deleteAndQueue('pigs', id, operation);
    setPigs((current) => current.filter((pig) => pig.id !== id));
    const updatedFeedLogs = feedLogs.map((entry) => (
      entry.pigId === id ? { ...entry, pigId: null } : entry
    ));
    await offlineStore.replaceRecords('feedLogs', updatedFeedLogs);
    setFeedLogs(updatedFeedLogs);
    setPendingSyncCount((count) => count + 1);
    void syncPendingChanges();
  }

  async function createFeedLog(entry) {
    const created = { ...entry, id: crypto.randomUUID() };
    const operation = offlineStore.createOperation('create', 'feedLogs', created.id, created);
    await offlineStore.saveAndQueue('feedLogs', created, operation);
    setFeedLogs((current) => [created, ...current]);
    setPendingSyncCount((count) => count + 1);
    void syncPendingChanges();
  }

  async function deleteFeedLog(id) {
    const operation = offlineStore.createOperation('delete', 'feedLogs', id);
    await offlineStore.deleteAndQueue('feedLogs', id, operation);
    setFeedLogs((current) => current.filter((entry) => entry.id !== id));
    setPendingSyncCount((count) => count + 1);
    void syncPendingChanges();
  }

  async function updateAlertStatus(id, status) {
    const alert = alerts.find((entry) => entry.id === id);
    if (!alert) throw new Error('The selected alert is not available in local storage.');
    const updated = { ...alert, status };
    const operation = offlineStore.createOperation('updateStatus', 'alerts', id, { status });
    await offlineStore.saveAndQueue('alerts', updated, operation);
    setAlerts((current) => current.map((entry) => entry.id === id ? updated : entry));
    setPendingSyncCount((count) => count + 1);
    void syncPendingChanges();
  }

  const createObservation = useCallback(async (detection) => {
    const preparedDetection = { ...detection, id: crypto.randomUUID() };
    const created = mapLocalDetection(preparedDetection, pigs);
    preparedDetection.pig_id = created.pigId || undefined;
    preparedDetection.pen_tag = created.pigId
      ? Number(created.penTag)
      : undefined;
    const operation = offlineStore.createOperation(
      'create',
      'observations',
      created.id,
      preparedDetection,
    );
    await offlineStore.saveAndQueue('observations', created, operation);
    setObservations((current) => [created, ...current]);
    if (created.pigId && created.weight != null) {
      setPigs((current) => current.map((pig) => (
        pig.id === created.pigId ? { ...pig, registeredWeight: created.weight } : pig
      )));
      const updatedPig = pigs.find((pig) => pig.id === created.pigId);
      if (updatedPig) {
        await offlineStore.saveRecord('pigs', { ...updatedPig, registeredWeight: created.weight });
      }
    }
    setPendingSyncCount((count) => count + 1);
    void syncPendingChanges();
  }, [pigs, syncPendingChanges]);

  const refreshDetectorStatus = useCallback(async () => {
    try {
      setDetectorStatus(await api.camera.status());
    } catch (error) {
      setDetectorStatus({ available: false, message: error.message || 'Could not refresh pig detector status.' });
    }
  }, []);

  function navigate(id) {
    if (id === 'menu') {
      setMobileMenuOpen(true);
      return;
    }
    setActivePage(id);
    setMobileMenuOpen(false);
  }

  if (splashVisible) {
    return <main className="splash-screen" aria-label="Loading Smart Swine dashboard">
      <div className="splash-content">
        <img className="splash-logo" src="/smart-swine-logo.svg" alt="Smart Swine pig logo" />
        <h1>Smart Swine</h1>
        <p>Farm management dashboard</p>
        <div className="splash-progress" role="progressbar" aria-label="Loading dashboard"><span /></div>
        <span className="splash-loading">Loading dashboard…</span>
      </div>
    </main>;
  }

  const PageIcon = page?.Icon ?? Database;
  return <div className="app-shell">
    <Sidebar activePage={activePage} onNavigate={navigate} pendingAlerts={alerts.filter((alert) => alert.status === 'Pending Review').length} databaseConnected={databaseConnected} />
    {mobileMenuOpen && <MobileMenu activePage={activePage} onNavigate={navigate} pendingAlerts={alerts.filter((alert) => alert.status === 'Pending Review').length} databaseConnected={databaseConnected} onClose={() => setMobileMenuOpen(false)} />}
    <div className="main-column">
      <header className="topbar"><div className="current-page"><PageIcon size={19} /><div><strong>{page?.label ?? 'Dashboard'}</strong><span>Smart Swine farm management</span></div></div><div className={`connection-status ${databaseConnected ? 'connected' : 'disconnected'}`}><span className={`status-dot ${databaseConnected ? '' : 'status-offline'}`} />{databaseStatus === 'connecting' ? 'Connecting…' : databaseConnected ? 'Supabase connected' : 'Offline · local storage'}{pendingSyncCount > 0 && ` · ${pendingSyncCount} pending`}</div></header>
      <div className="system-strip"><span className={databaseConnected ? '' : 'disconnected'}>{databaseConnected ? <Database /> : <AlertCircle />}{databaseConnected ? 'Supabase connected' : databaseStatus === 'connecting' ? 'Checking connection' : 'Local data available'}</span><span className={detectorStatus.available ? '' : 'disconnected'} title={detectorStatus.message}><Cpu />{detectorStatus.available ? 'Pig detector ready' : 'Pig detector unavailable'}</span><span><Scale />{pigs.length} registered animals</span></div>
      <main className="content">
        {!databaseConnected && databaseStatus !== 'connecting' && <div className="database-error local-storage-notice" role="status"><AlertCircle size={18} /><div><strong>Offline mode · local storage</strong><p>{databaseError || 'Your farm records remain available on this device. New changes will sync automatically when Supabase is reachable.'}</p>{syncError && <p>Sync paused: {syncError}</p>}<button className="text-button" onClick={() => void refreshAndSync()} disabled={dataLoading}>Retry sync</button></div></div>}
        {databaseConnected && pendingSyncCount > 0 && <div className="database-error local-storage-notice" role="status"><Database size={18} /><div><strong>Local changes waiting to sync</strong><p>{syncError || `${pendingSyncCount} change${pendingSyncCount === 1 ? '' : 's'} waiting to upload to Supabase.`}</p><button className="text-button" onClick={() => void syncPendingChanges()}>Sync now</button></div></div>}
        <div hidden={activePage !== 'live'}>
          <LiveMonitor pigs={pigs} observations={observations} onRefresh={refreshAndSync} loading={dataLoading} offline={!databaseConnected} detectorAvailable={detectorStatus.available} detectorMessage={detectorStatus.message} postureStatus={detectorStatus.posture} onRecordDetection={createObservation} />
        </div>
        {activePage === 'records' && <PigRegistry pigs={pigs} onCreate={createPig} onDelete={deletePig} offline={!databaseConnected} />}
        {activePage === 'reports' && <Suspense fallback={<div className="panel empty-state">Loading analytics…</div>}><Analytics pigs={pigs} observations={observations} offline={false} /></Suspense>}
        {activePage === 'feed' && <FeedExpenses pigs={pigs} feedLogs={feedLogs} onCreate={createFeedLog} onDelete={deleteFeedLog} offline={!databaseConnected} />}
        {activePage === 'alerts' && <HealthAlerts alerts={alerts} onUpdateStatus={updateAlertStatus} offline={!databaseConnected} />}
        {activePage === 'settings' && <Settings databaseStatus={databaseStatus} onRefresh={refreshAndSync} loading={dataLoading} onDetectorUpdated={() => void refreshDetectorStatus()} />}
      </main>
      <footer className="app-footer">Farm records are stored on this device and sync to Supabase when connected.</footer>
    </div>
  </div>;
}
