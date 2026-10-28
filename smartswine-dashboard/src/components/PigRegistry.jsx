import { useMemo, useState } from 'react';
import { Database, Plus, Search, Trash2, X } from 'lucide-react';

const emptyForm = { aiIdentity: '', penTag: '', stage: 'Grower', registeredWeight: '', status: 'Active', notes: '' };

function createPigForm(pigs) {
  const identities = new Set(pigs.map((pig) => pig.aiIdentity?.replace(/[^a-z0-9]/gi, '').toLowerCase()));
  const penCounts = new Map();
  for (const pig of pigs) {
    const tag = Number(pig.penTag);
    penCounts.set(tag, (penCounts.get(tag) || 0) + 1);
  }
  let identityNumber = 1;
  while (identities.has(`pig${identityNumber}`)) identityNumber += 1;
  let penTag = 1;
  while ((penCounts.get(penTag) || 0) >= 20) penTag += 1;

  return { ...emptyForm, aiIdentity: `Pig_${identityNumber}`, penTag: String(penTag) };
}

export default function PigRegistry({ pigs, onCreate, onDelete, offline = false }) {
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');
  const [selectedPen, setSelectedPen] = useState('');
  const penTags = useMemo(
    () => [...new Set(pigs.map((pig) => String(Number(pig.penTag))))].sort((a, b) => Number(a) - Number(b)),
    [pigs],
  );
  const visiblePigs = useMemo(() => {
    const query = search.trim().toLowerCase();
    return pigs.filter((pig) => {
      const matchesSearch = !query || [
        pig.aiIdentity,
        pig.penTag,
        pig.stage,
        pig.status,
        pig.notes,
      ].some((value) => String(value || '').toLowerCase().includes(query));
      return matchesSearch && (!selectedPen || Number(pig.penTag) === Number(selectedPen));
    });
  }, [pigs, search, selectedPen]);

  async function savePig(event) {
    event.preventDefault();
    const tagNumber = Number(form.penTag);
    if (!Number.isInteger(tagNumber) || tagNumber < 1 || tagNumber > 2147483647) {
      setError('Enter a whole-number pen tag from 1 to 2,147,483,647.');
      return;
    }
    const penTag = String(tagNumber).padStart(2, '0');
    const pigsInPen = pigs.filter((pig) => Number(pig.penTag) === tagNumber).length;
    if (pigsInPen >= 20) {
      setError(`Pen ${penTag} already has 20 pigs. Choose another pen.`);
      return;
    }
    const identity = form.aiIdentity.trim();
    const identityKey = identity.replace(/[^a-z0-9]/gi, '').toLowerCase();
    if (identityKey && pigs.some((pig) => (
      Number(pig.penTag) === tagNumber
      &&
      pig.aiIdentity?.replace(/[^a-z0-9]/gi, '').toLowerCase() === identityKey
    ))) {
      setError(`AI identity ${identity} is already assigned to pen ${penTag}.`);
      return;
    }
    const weight = Number(form.registeredWeight);
    if (!Number.isFinite(weight) || weight < 0) {
      setError('Enter a valid non-negative registered weight.');
      return;
    }
    setSaving(true);
    try {
      await onCreate({ ...form, aiIdentity: identity || null, penTag, registeredWeight: weight, notes: form.notes.trim() });
      setForm(createPigForm(pigs));
      setError('');
      setShowForm(false);
    } catch (saveError) {
      setError(saveError.message);
    } finally {
      setSaving(false);
    }
  }

  async function removePig(id) {
    setError('');
    try {
      await onDelete(id);
    } catch (deleteError) {
      setError(deleteError.message);
    }
  }

  return (
    <div className="page-stack">
      <section className="panel page-title-row"><div><div className="title-with-icon"><Database className="accent-icon" /><h2>Farm herd registry</h2></div><p>Register up to 20 pigs in each pen.</p></div><button className="button button-primary" onClick={() => { setError(''); setForm(createPigForm(pigs)); setShowForm(true); }}><Plus size={17} />Register pig</button></section>
      <section className="panel registry-filters" aria-label="Search and filter pig registry">
        <label className="registry-search">
          <Search size={17} aria-hidden="true" />
          <input
            type="search"
            aria-label="Search pigs"
            placeholder="Search by identity, pen tag, stage, or notes"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
        <label className="registry-pen-filter">
          <span>Show pen</span>
          <select aria-label="Filter by pen tag" value={selectedPen} onChange={(event) => setSelectedPen(event.target.value)}>
            <option value="">All pens</option>
            {penTags.map((tag) => <option value={tag} key={tag}>Pen #{String(tag).padStart(2, '0')}</option>)}
          </select>
        </label>
        <span className="registry-result-count" aria-live="polite">
          Showing {visiblePigs.length} of {pigs.length} pigs
          {selectedPen && ` · Pen #${String(selectedPen).padStart(2, '0')}`}
        </span>
      </section>
      <div className="card-grid">
        {visiblePigs.map((pig) => <article className="panel pig-card" key={pig.id}>
          <div className="row-between"><span className="identity">{pig.aiIdentity}</span><span className={`pill ${pig.status === 'Active' ? 'pill-green' : 'pill-red'}`}>{pig.status}</span></div>
          <div><small className="muted">Pen tag</small><h3 className="pen-tag">#{pig.penTag}</h3></div>
          <div className="two-column"><div><small className="muted">Growth stage</small><strong>{pig.stage}</strong></div><div><small className="muted">Latest weight</small><strong>{Number(pig.registeredWeight).toFixed(1)} kg</strong></div></div>
          <p className="pig-notes">{pig.notes}</p>
          <button className="text-button danger-text" onClick={() => removePig(pig.id)}><Trash2 size={15} />Remove</button>
        </article>)}
      </div>
      {error && !showForm && <p className="inline-error panel">{error}</p>}
      {!pigs.length && <p className="empty-state panel">No pigs are saved on this device yet. Register a pig to get started.</p>}
      {pigs.length > 0 && visiblePigs.length === 0 && <p className="empty-state panel">No pigs match your search and pen filter.</p>}
      {showForm && <div className="modal-backdrop" onClick={() => setShowForm(false)}><section className="modal panel" role="dialog" aria-modal="true" aria-labelledby="pig-form-title" onClick={(event) => event.stopPropagation()}>
        <div className="row-between modal-header"><h2 id="pig-form-title">Register pig profile</h2><button className="icon-button" onClick={() => setShowForm(false)} aria-label="Close"><X /></button></div>
        <form className="form-grid" onSubmit={savePig}>
          <label>AI identity<input maxLength="50" value={form.aiIdentity} onChange={(e) => setForm({ ...form, aiIdentity: e.target.value })} /></label>
          <label>Pen tag<input required type="number" min="1" max="2147483647" step="1" value={form.penTag} onChange={(e) => setForm({ ...form, penTag: e.target.value })} /></label>
          <label>Growth stage<select value={form.stage} onChange={(e) => setForm({ ...form, stage: e.target.value })}>{['Nursery', 'Grower', 'Fattening', 'Sow'].map((stage) => <option key={stage}>{stage}</option>)}</select></label>
          <label>Weight (kg)<input required type="number" min="0" step="0.1" value={form.registeredWeight} onChange={(e) => setForm({ ...form, registeredWeight: e.target.value })} /></label>
          <label className="form-full">Notes<input value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} /></label>
          {error && <p className="inline-error form-full">{error}</p>}
          <div className="form-actions form-full"><button type="button" className="button button-secondary" onClick={() => setShowForm(false)}>Cancel</button><button className="button button-primary" type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save profile'}</button></div>
        </form>
      </section></div>}
    </div>
  );
}
