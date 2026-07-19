import { motion } from 'framer-motion'
import { Dna, ScanLine, BarChart3, Shield, Brain, Zap } from 'lucide-react'
import { Link } from 'react-router-dom'

const features = [
  {
    icon: ScanLine,
    title: 'YOLOv8 Detection',
    description: 'Real-time lesion detection with bounding box localization',
    color: 'var(--accent-primary)',
  },
  {
    icon: Brain,
    title: 'Attention U-Net',
    description: 'Precise tumor segmentation with attention-gated skip connections',
    color: 'var(--accent-secondary)',
  },
  {
    icon: BarChart3,
    title: 'Ensemble Classification',
    description: 'ResNet50 + EfficientNetV2 soft voting for maximum accuracy',
    color: 'var(--accent-success)',
  },
  {
    icon: Shield,
    title: 'Grad-CAM++ XAI',
    description: 'Visual heatmaps explaining exactly what the AI detected',
    color: 'var(--accent-warning)',
  },
]

const stats = [
  { label: 'Pipeline Phases', value: '4', suffix: 'stages' },
  { label: 'AI Models', value: '5', suffix: 'trained' },
  { label: 'Target Accuracy', value: '97', suffix: '%+' },
  { label: 'Inference Time', value: '<200', suffix: 'ms' },
]

export default function HomePage() {
  return (
    <div>
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        style={{ marginBottom: 'var(--space-3xl)' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
          <Dna size={36} color="var(--accent-primary)" />
          <div>
            <h1 style={{ fontSize: '2rem', fontWeight: 800, lineHeight: 1.2 }}>
              Ultra<span style={{ color: 'var(--accent-primary)' }}>X</span>AI
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Explainable AI for Breast Cancer Detection
            </p>
          </div>
        </div>

        <p style={{
          fontSize: '1.1rem',
          color: 'var(--text-secondary)',
          maxWidth: '700px',
          lineHeight: 1.7,
        }}>
          A multi-model hybrid ensemble architecture that <strong style={{ color: 'var(--text-primary)' }}>detects</strong>,{' '}
          <strong style={{ color: 'var(--text-primary)' }}>segments</strong>,{' '}
          <strong style={{ color: 'var(--text-primary)' }}>classifies</strong>, and{' '}
          <strong style={{ color: 'var(--text-primary)' }}>explains</strong>{' '}
          breast cancer findings from ultrasound imagery with clinical-grade precision.
        </p>

        <Link to="/diagnose">
          <motion.button
            className="btn btn-primary btn-lg"
            style={{ marginTop: 'var(--space-xl)' }}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
          >
            <Zap size={18} />
            Start AI Diagnosis
          </motion.button>
        </Link>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        className="stats-grid"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        style={{ marginBottom: 'var(--space-2xl)' }}
      >
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            className="stat-card"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + i * 0.1 }}
          >
            <p className="stat-card-label">{stat.label}</p>
            <p className="stat-card-value">
              {stat.value}
              <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginLeft: '0.25rem' }}>
                {stat.suffix}
              </span>
            </p>
          </motion.div>
        ))}
      </motion.div>

      {/* 4-Phase Pipeline Visualization */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: 'var(--space-lg)' }}>
          🧬 4-Phase AI Pipeline
        </h2>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: 'var(--space-lg)',
        }}>
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              className="glass-card"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.12 }}
              style={{ cursor: 'default' }}
            >
              <div style={{
                width: 42,
                height: 42,
                borderRadius: 'var(--radius-md)',
                background: `${feature.color}15`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 'var(--space-md)',
              }}>
                <feature.icon size={22} color={feature.color} />
              </div>

              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 'var(--space-xs)' }}>
                Phase {i + 1}: {feature.title}
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Datasets Info */}
      <motion.div
        className="glass-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
        style={{ marginTop: 'var(--space-2xl)', cursor: 'default' }}
      >
        <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 'var(--space-md)' }}>
          📊 Training Datasets
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 'var(--space-md)' }}>
          {[
            { name: 'BUSI', count: '780', masks: true },
            { name: 'BUS-BRA', count: '1,875', masks: true },
            { name: 'UDIAT', count: '163', masks: true },
            { name: 'STU', count: '756', masks: true },
            { name: 'OASBUD', count: '200', masks: false },
          ].map(ds => (
            <div key={ds.name} style={{
              background: 'var(--bg-tertiary)',
              borderRadius: 'var(--radius-md)',
              padding: 'var(--space-md)',
              textAlign: 'center',
            }}>
              <p style={{ fontWeight: 700, fontSize: '0.9rem' }}>{ds.name}</p>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem', color: 'var(--accent-primary)', fontWeight: 700 }}>{ds.count}</p>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
                {ds.masks ? '✅ Masks' : '❌ No Masks'}
              </p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
