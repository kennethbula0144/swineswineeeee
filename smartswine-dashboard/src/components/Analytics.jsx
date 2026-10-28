import { Activity, BarChart3, Scale } from 'lucide-react';
import { CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const chartColors = ['#34d399', '#60a5fa', '#fbbf24', '#fb7185', '#c084fc', '#22d3ee'];

export default function Analytics({ pigs, observations, offline = false }) {
  const totalWeight = pigs.reduce((sum, pig) => sum + Number(pig.registeredWeight || 0), 0);
  const averageWeight = pigs.length ? totalWeight / pigs.length : null;
  const behaviorMap = observations.reduce((counts, item) => {
    if (item.behavior) counts[item.behavior] = (counts[item.behavior] || 0) + 1;
    return counts;
  }, {});
  const behaviorData = Object.entries(behaviorMap).map(([name, value]) => ({ name, value }));
  const identities = [...new Set(observations.map((item) => item.aiIdentity).filter(Boolean))];
  const dailyWeights = observations.reduce((days, item) => {
    if (item.weight == null || !item.createdAt || !item.aiIdentity) return days;
    const day = item.createdAt.slice(0, 10);
    const entry = days.get(day) || { day };
    entry[item.aiIdentity] = Number(item.weight);
    days.set(day, entry);
    return days;
  }, new Map());
  const weightData = [...dailyWeights.values()].sort((left, right) => left.day.localeCompare(right.day));

  return <div className="page-stack">
    <div className="metric-grid">
      <Metric label="Registered herd" value={`${pigs.length} head`} detail="Locally stored pig records" color="green" />
      <Metric label="Total registered weight" value={`${totalWeight.toFixed(1)} kg`} detail="Combined registered pig weights" color="rose" />
      <Metric label="Average registered weight" value={averageWeight == null ? '—' : `${averageWeight.toFixed(1)} kg`} detail="Calculated from saved records" color="blue" />
      <Metric label="Observations" value={observations.length} detail="Locally stored observation records" color="amber" />
      <Metric label="Pen tags used" value={new Set(pigs.map((pig) => pig.penTag)).size} detail="Unique pen assignments" color="rose" />
    </div>
    <div className="chart-grid">
      <section className="panel chart-panel"><h2><Scale className="accent-icon" />Recorded weight observations</h2><p className="chart-note">Weights from saved observations, grouped by observation date.</p>{weightData.length ? <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={weightData}><CartesianGrid stroke="#26354a" strokeDasharray="4 4" /><XAxis dataKey="day" stroke="#91a2b8" /><YAxis stroke="#91a2b8" /><Tooltip contentStyle={{ background: '#121d2d', border: '1px solid #334155', borderRadius: 10 }} /><Legend />{identities.map((id, index) => <Line key={id} type="monotone" dataKey={id} stroke={chartColors[index % chartColors.length]} strokeWidth={2} dot={false} connectNulls />)}</LineChart></ResponsiveContainer></div> : <div className="empty-chart"><BarChart3 />No recorded weight observations.</div>}</section>
      <section className="panel chart-panel"><h2><Activity className="accent-icon" />Recorded behavior mix</h2><p className="chart-note">Calculated from saved observations.</p>{behaviorData.length ? <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={behaviorData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={4}>{behaviorData.map((item, index) => <Cell key={item.name} fill={chartColors[index % chartColors.length]} />)}</Pie><Tooltip contentStyle={{ background: '#121d2d', border: '1px solid #334155', borderRadius: 10 }} /><Legend /></PieChart></ResponsiveContainer></div> : <div className="empty-chart"><BarChart3 />No recorded behaviors.</div>}</section>
    </div>
  </div>;
}

function Metric({ label, value, detail, color }) {
  return <section className="panel metric-card"><small>{label}</small><strong className={`metric-value ${color}`}>{value}</strong><span>{detail}</span></section>;
}
