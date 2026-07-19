import { NavLink } from 'react-router-dom'
import { Home, ScanLine, BarChart3, Activity, Dna } from 'lucide-react'

const navItems = [
  { path: '/', label: 'Dashboard', icon: Home },
  { path: '/diagnose', label: 'AI Diagnosis', icon: ScanLine },
  { path: '/benchmark', label: 'Benchmarks', icon: BarChart3 },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <Dna size={28} color="var(--accent-primary)" />
        <div>
          <h1>UltraXAI</h1>
          <span>Breast Cancer Detection</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `sidebar-link ${isActive ? 'active' : ''}`
            }
          >
            <item.icon size={20} />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div style={{ marginTop: 'auto', paddingTop: 'var(--space-xl)' }}>
        <div className="glass-card" style={{
          padding: 'var(--space-md)',
          textAlign: 'center',
          background: 'var(--accent-primary-dim)',
          border: '1px solid var(--border-accent)',
        }}>
          <Activity size={16} color="var(--accent-primary)" style={{ margin: '0 auto 0.5rem' }} />
          <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>System Status</p>
          <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-success)' }}>● Online</p>
        </div>
      </div>
    </aside>
  )
}
