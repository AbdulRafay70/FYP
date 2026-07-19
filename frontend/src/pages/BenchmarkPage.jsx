import { motion } from 'framer-motion'
import { BarChart3, Clock, Target, Activity } from 'lucide-react'

const trackAMetrics = {
  name: 'Track A — Clinical Pipeline',
  subtitle: 'YOLO → Attention U-Net → ResNet50 + EfficientNetV2',
  color: 'var(--accent-primary)',
  metrics: [
    { label: 'Accuracy', value: '—', target: '> 97%' },
    { label: 'Sensitivity', value: '—', target: '> 98%' },
    { label: 'Specificity', value: '—', target: '> 95%' },
    { label: 'F1 Score', value: '—', target: '> 97%' },
    { label: 'AUC-ROC', value: '—', target: '> 0.99' },
    { label: 'Dice (Seg)', value: '—', target: '> 0.85' },
    { label: 'IoU (Seg)', value: '—', target: '> 0.80' },
    { label: 'Inference', value: '—', target: '< 200ms' },
  ],
}

const trackBMetrics = {
  name: 'Track B — SOTA Research Pipeline',
  subtitle: 'MedSAM → Swin Transformer V2 + ConvNeXt V2',
  color: 'var(--accent-secondary)',
  metrics: [
    { label: 'Accuracy', value: '—', target: '> 98%' },
    { label: 'Sensitivity', value: '—', target: '> 99%' },
    { label: 'Specificity', value: '—', target: '> 96%' },
    { label: 'F1 Score', value: '—', target: '> 98%' },
    { label: 'AUC-ROC', value: '—', target: '> 0.99' },
    { label: 'Dice (Seg)', value: '—', target: '> 0.90' },
    { label: 'IoU (Seg)', value: '—', target: '> 0.85' },
    { label: 'Inference', value: '—', target: '< 800ms' },
  ],
}

function MetricsCard({ track, delay = 0 }) {
  return (
    <motion.div
      className="glass-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      style={{ cursor: 'default' }}
    >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-md)',
        marginBottom: 'var(--space-lg)',
        paddingBottom: 'var(--space-md)',
        borderBottom: '1px solid var(--border-default)',
      }}>
        <div style={{
          width: 40,
          height: 40,
          borderRadius: 'var(--radius-md)',
          background: `${track.color}15`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <BarChart3 size={20} color={track.color} />
        </div>
        <div>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>{track.name}</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>{track.subtitle}</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-sm)' }}>
        {track.metrics.map((metric) => (
          <div key={metric.label} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: 'var(--space-sm) var(--space-md)',
            background: 'var(--bg-tertiary)',
            borderRadius: 'var(--radius-sm)',
          }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {metric.label}
            </span>
            <div style={{ textAlign: 'right' }}>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                fontSize: '0.9rem',
                color: metric.value === '—' ? 'var(--text-tertiary)' : 'var(--text-primary)',
              }}>
                {metric.value}
              </span>
              <p style={{ fontSize: '0.6rem', color: 'var(--text-tertiary)' }}>
                Target: {metric.target}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div style={{
        marginTop: 'var(--space-lg)',
        padding: 'var(--space-sm) var(--space-md)',
        borderRadius: 'var(--radius-md)',
        background: 'var(--accent-warning-dim)',
        border: '1px solid rgba(245, 158, 11, 0.2)',
      }}>
        <p style={{ fontSize: '0.75rem', color: 'var(--accent-warning)' }}>
          ⏳ Awaiting training — Metrics will update after model training is complete
        </p>
      </div>
    </motion.div>
  )
}

export default function BenchmarkPage() {
  return (
    <div>
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>📊 Model Benchmarks</h1>
        <p>Compare Track A (Clinical) vs Track B (SOTA) pipeline performance</p>
      </motion.div>

      {/* Overview Stats */}
      <motion.div
        className="stats-grid"
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        style={{ marginBottom: 'var(--space-2xl)' }}
      >
        {[
          { icon: Target, label: 'Total Models', value: '7', color: 'var(--accent-primary)' },
          { icon: Activity, label: 'Datasets', value: '5', color: 'var(--accent-success)' },
          { icon: BarChart3, label: 'Total Images', value: '3,574', color: 'var(--accent-secondary)' },
          { icon: Clock, label: 'Cross-Val Folds', value: '5', color: 'var(--accent-warning)' },
        ].map((stat, i) => (
          <motion.div
            key={stat.label}
            className="stat-card"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 + i * 0.08 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-xs)' }}>
              <stat.icon size={16} color={stat.color} />
              <p className="stat-card-label" style={{ margin: 0 }}>{stat.label}</p>
            </div>
            <p className="stat-card-value">{stat.value}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* Track Comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
        <MetricsCard track={trackAMetrics} delay={0.2} />
        <MetricsCard track={trackBMetrics} delay={0.3} />
      </div>

      {/* Evaluation Methodology */}
      <motion.div
        className="glass-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        style={{ marginTop: 'var(--space-xl)', cursor: 'default' }}
      >
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 'var(--space-md)' }}>
          🧪 Evaluation Methodology
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-lg)' }}>
          <div>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-primary)', marginBottom: 'var(--space-sm)' }}>
              Segmentation Metrics
            </h4>
            <ul style={{ listStyle: 'none', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <li style={{ marginBottom: '0.25rem' }}>• Dice Similarity Coefficient (DSC)</li>
              <li style={{ marginBottom: '0.25rem' }}>• Intersection over Union (IoU)</li>
              <li style={{ marginBottom: '0.25rem' }}>• Hausdorff Distance</li>
              <li>• Pixel Accuracy</li>
            </ul>
          </div>
          <div>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-primary)', marginBottom: 'var(--space-sm)' }}>
              Classification Metrics
            </h4>
            <ul style={{ listStyle: 'none', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <li style={{ marginBottom: '0.25rem' }}>• Accuracy, Sensitivity, Specificity</li>
              <li style={{ marginBottom: '0.25rem' }}>• F1-Score, Precision, NPV</li>
              <li style={{ marginBottom: '0.25rem' }}>• AUC-ROC & Precision-Recall Curves</li>
              <li>• McNemar's Statistical Test (Track A vs B)</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
