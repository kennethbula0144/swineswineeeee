import { useState } from 'react';
import { Banknote, Trash2 } from 'lucide-react';

export default function FeedExpenses({ pigs, feedLogs, onCreate, onDelete, offline = false }) {
  const [form, setForm] = useState({ pigId: pigs[0]?.id ?? '', feedType: '', quantityKg: '', costPerKg: '' });
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const total = feedLogs.reduce((sum, item) => sum + Number(item.quantityKg) * Number(item.costPerKg), 0);
  const formatCurrency = (amount) => new Intl.NumberFormat('en-PH', {
    style: 'currency',
    currency: 'PHP',
  }).format(amount);

  async function addEntry(event) {
    event.preventDefault();
    const quantityKg = Number(form.quantityKg);
    const costPerKg = Number(form.costPerKg);
    const pig = pigs.find((entry) => entry.id === form.pigId);
    if (!pig || !form.feedType.trim() || !Number.isFinite(quantityKg) || quantityKg <= 0 || !Number.isFinite(costPerKg) || costPerKg < 0) {
      setError('Choose a pen, enter a feed type, and provide a valid quantity and cost.');
      return;
    }
    setSaving(true);
    try {
      await onCreate({ pigId: pig.id, penTag: pig.penTag, feedType: form.feedType.trim(), quantityKg, costPerKg, date: new Date().toISOString().slice(0, 10) });
      setForm({ ...form, quantityKg: '' });
      setError('');
    } catch (saveError) {
      setError(saveError.message);
    } finally {
      setSaving(false);
    }
  }

  async function removeEntry(id) {
    setError('');
    try {
      await onDelete(id);
    } catch (deleteError) {
      setError(deleteError.message);
    }
  }

  return <div className="feed-grid">
    <section className="panel feed-form-panel"><h2><Banknote className="accent-icon" />Log feed entry</h2><form className="form-grid single-column" onSubmit={addEntry}><fieldset className="offline-fieldset">
      <label>Pen tag<select required value={form.pigId} onChange={(e) => setForm({ ...form, pigId: e.target.value })}><option value="">Select a pig</option>{pigs.map((pig) => <option value={pig.id} key={pig.id}>#{pig.penTag} · {pig.stage}</option>)}</select></label>
      <label>Feed type<input required value={form.feedType} onChange={(e) => setForm({ ...form, feedType: e.target.value })} /></label>
      <label>Quantity (kg)<input required min="0.01" step="0.01" type="number" value={form.quantityKg} onChange={(e) => setForm({ ...form, quantityKg: e.target.value })} /></label>
      <label>Cost per kg (₱)<input required min="0" step="0.01" type="number" value={form.costPerKg} onChange={(e) => setForm({ ...form, costPerKg: e.target.value })} /></label>
      {error && <p className="inline-error">{error}</p>}<button className="button button-primary" disabled={!pigs.length || saving}>{saving ? 'Saving…' : 'Add feed record'}</button>
      {offline && <p className="muted">Saved on this device and queued to sync when connected.</p>}
      {!pigs.length && <p className="muted">Register a pig before adding feed records.</p>}
    </fieldset></form></section>
    <section className="panel table-panel"><div className="row-between table-heading"><div><h2>Feed consumption & expenses</h2><p>{offline ? 'Showing records saved on this device.' : 'Records sync with the Supabase database.'}</p></div><strong className="total-cost">{formatCurrency(total)}</strong></div>
      <div className="table-wrap"><table><thead><tr><th>Date</th><th>Pen tag</th><th>Feed</th><th>Quantity</th><th>Cost/kg</th><th>Total</th><th /></tr></thead><tbody>{feedLogs.map((item) => <tr key={item.id}><td>{item.date}</td><td>#{item.penTag}</td><td>{item.feedType}</td><td>{Number(item.quantityKg).toFixed(1)} kg</td><td>{formatCurrency(Number(item.costPerKg))}</td><td>{formatCurrency(Number(item.quantityKg) * Number(item.costPerKg))}</td><td><button className="icon-button danger-text" aria-label={`Delete feed entry for ${item.feedType}`} onClick={() => removeEntry(item.id)}><Trash2 size={15} /></button></td></tr>)}</tbody></table>{!feedLogs.length && <p className="empty-state">No feed records have been logged.</p>}</div>
    </section>
  </div>;
}
