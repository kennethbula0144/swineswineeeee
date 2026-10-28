import { AlertTriangle, BarChart3, Camera, Database, DollarSign, Menu, Settings, X } from 'lucide-react';

export const pages = [
  { id: 'live', label: 'Live pen monitor', Icon: Camera },
  { id: 'records', label: 'Pig registry', Icon: Database },
  { id: 'reports', label: 'Analytics & trends', Icon: BarChart3 },
  { id: 'feed', label: 'Feed & expenses', Icon: DollarSign },
  { id: 'alerts', label: 'Health alerts', Icon: AlertTriangle },
  { id: 'settings', label: 'Settings & training', Icon: Settings },
];

export default function Sidebar({ activePage, onNavigate, pendingAlerts, databaseConnected }) {
  return (
    <>
      <header className="mobile-header">
        <button className="icon-button" onClick={() => onNavigate('menu')} aria-label="Open navigation"><Menu /></button>
        <Brand />
      </header>
      <aside className="sidebar">
        <Brand />
        <nav className="nav-list" aria-label="Main navigation">
          {pages.map(({ id, label, Icon }) => (
            <button className={`nav-item ${activePage === id ? 'selected' : ''}`} key={id} onClick={() => onNavigate(id)}>
              <Icon size={19} />
              <span>{label}</span>
              {id === 'alerts' && pendingAlerts > 0 && <span className="nav-badge">{pendingAlerts}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-note"><span className={`status-dot ${databaseConnected ? '' : 'status-offline'}`} />{databaseConnected ? 'Supabase connected' : 'Local storage mode'}</div>
      </aside>
    </>
  );
}

export function MobileMenu({ activePage, onNavigate, pendingAlerts, databaseConnected, onClose }) {
  return (
    <div className="mobile-backdrop" onClick={onClose}>
      <aside className="mobile-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-top"><Brand /><button className="icon-button" onClick={onClose} aria-label="Close navigation"><X /></button></div>
        <nav className="nav-list" aria-label="Mobile navigation">
          {pages.map(({ id, label, Icon }) => (
            <button className={`nav-item ${activePage === id ? 'selected' : ''}`} key={id} onClick={() => onNavigate(id)}>
              <Icon size={19} /><span>{label}</span>
              {id === 'alerts' && pendingAlerts > 0 && <span className="nav-badge">{pendingAlerts}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-note"><span className={`status-dot ${databaseConnected ? '' : 'status-offline'}`} />{databaseConnected ? 'Supabase connected' : 'Local storage mode'}</div>
      </aside>
    </div>
  );
}

function Brand() {
  return <div className="brand"><img className="brand-logo" src="/smart-swine-logo.svg" alt="" /><span><strong>Smart Swine</strong><small>Farm dashboard</small></span></div>;
}
