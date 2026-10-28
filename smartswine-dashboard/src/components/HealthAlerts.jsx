import { ShieldAlert, UserCheck } from 'lucide-react';

import { useState } from 'react';

export default function HealthAlerts({ alerts, onUpdateStatus, offline = false }) {
  const [error, setError] = useState('');
  const [updating, setUpdating] = useState('');
  async function toggleStatus(alert) {
    setError('');
    setUpdating(alert.id);
    try {
      await onUpdateStatus(alert.id, alert.status === 'Pending Review' ? 'Referred to Vet' : 'Pending Review');
    } catch (updateError) {
      setError(updateError.message);
    } finally {
      setUpdating('');
    }
  }
  return <div className="page-stack">
    <div className="notice"><ShieldAlert /><div><strong>Health review</strong><p>{offline ? 'Changes are saved on this device and will sync when connected. Contact a veterinarian for animal health concerns.' : 'Use these records as decision-support information. Contact a veterinarian for animal health concerns.'}</p></div></div>
    {error && <p className="inline-error panel">{error}</p>}
    {alerts.map((alert) => <article className="panel alert-card" key={alert.id}><div className="row-between alert-heading"><div><span className={`pill ${alert.severity === 'High' || alert.severity === 'Critical' ? 'pill-red' : 'pill-amber'}`}>{alert.severity} severity</span><h2>{alert.issue}</h2></div><time>{alert.timestamp}</time></div><div className="alert-meta"><span>Identity <b>{alert.aiIdentity}</b></span><span>Pen <b>#{alert.penTag}</b></span><span>Status <b className={alert.status === 'Referred to Vet' ? 'blue' : 'amber'}>{alert.status}</b></span></div><p className="alert-details">{alert.details}</p><div className="alert-action"><button className="button button-secondary" disabled={updating === alert.id} onClick={() => toggleStatus(alert)}><UserCheck size={16} />{updating === alert.id ? 'Saving…' : alert.status === 'Pending Review' ? 'Mark referred' : 'Return to review'}</button></div></article>)}
    {!alerts.length && <div className="panel empty-state">No health alerts have been recorded.</div>}
  </div>;
}
