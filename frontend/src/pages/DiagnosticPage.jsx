import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Image as ImageIcon, CheckCircle2, AlertCircle, Clock, Loader2, ShieldCheck, ShieldAlert, ShieldQuestion } from 'lucide-react'
import axios from 'axios'

const API_URL = 'http://localhost:8000/api/v1'

const PHASES = [
  { id: 1, name: 'Detection', description: 'YOLOv8-Seg' },
  { id: 2, name: 'Segmentation', description: 'Attention U-Net' },
  { id: 3, name: 'Classification', description: 'Ensemble' },
  { id: 4, name: 'Explainability', description: 'Grad-CAM++' },
]

export default function DiagnosticPage() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentPhase, setCurrentPhase] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileSelect = useCallback((file) => {
    if (!file) return

    const validExts = ['.png', '.jpg', '.jpeg', '.bmp']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!validExts.includes(ext)) {
      setError(`Unsupported format: ${ext}. Use PNG, JPG, or BMP.`)
      return
    }

    setSelectedFile(file)
    setError(null)
    setResult(null)

    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFileSelect(file)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const runDiagnosis = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setCurrentPhase(1)
    setError(null)
    setResult(null)

    // Simulate phase progression for visual feedback
    const phaseTimer = setInterval(() => {
      setCurrentPhase(prev => prev < 4 ? prev + 1 : prev)
    }, 600)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await axios.post(`${API_URL}/diagnose`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      })

      clearInterval(phaseTimer)
      setCurrentPhase(4)
      setResult(response.data)
    } catch (err) {
      clearInterval(phaseTimer)
      const msg = err.response?.data?.detail || err.message || 'Connection failed. Is the backend running?'
      setError(msg)
    } finally {
      setIsProcessing(false)
    }
  }

  const resetDiagnosis = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setCurrentPhase(0)
  }

  const getTrustIcon = (score) => {
    switch (score) {
      case 'HIGH': return <ShieldCheck size={20} color="var(--accent-success)" />
      case 'MEDIUM': return <ShieldQuestion size={20} color="var(--accent-warning)" />
      case 'LOW': return <ShieldAlert size={20} color="var(--accent-danger)" />
      default: return null
    }
  }

  const getTrustBadge = (score) => {
    const classes = {
      'HIGH': 'badge-success',
      'MEDIUM': 'badge-warning',
      'LOW': 'badge-danger',
    }
    return classes[score] || 'badge-info'
  }

  return (
    <div>
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>🔬 AI Diagnosis</h1>
        <p>Upload a breast ultrasound image for automated analysis</p>
      </motion.div>

      {/* Upload Zone */}
      {!preview && (
        <motion.div
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="upload-zone-content">
            <Upload className="upload-zone-icon" />
            <h3>Drop ultrasound image here</h3>
            <p>or click to browse — Supports PNG, JPG, BMP</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFileSelect(e.target.files[0])}
          />
        </motion.div>
      )}

      {/* Preview & Controls */}
      {preview && !result && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{ display: 'flex', gap: 'var(--space-xl)', alignItems: 'flex-start' }}
        >
          <div className="glass-card" style={{ flex: '0 0 300px', padding: 'var(--space-md)', cursor: 'default' }}>
            <img src={preview} alt="Uploaded scan" style={{ borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-md)' }} />
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
              {selectedFile?.name}
            </p>
          </div>

          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', gap: 'var(--space-md)', marginBottom: 'var(--space-xl)' }}>
              <motion.button
                className="btn btn-primary btn-lg"
                onClick={runDiagnosis}
                disabled={isProcessing}
                whileHover={!isProcessing ? { scale: 1.03 } : {}}
                whileTap={!isProcessing ? { scale: 0.98 } : {}}
              >
                {isProcessing ? (
                  <>
                    <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                    Processing...
                  </>
                ) : (
                  <>
                    <ImageIcon size={18} />
                    Run AI Diagnosis
                  </>
                )}
              </motion.button>

              <button className="btn btn-outline" onClick={resetDiagnosis}>
                Reset
              </button>
            </div>

            {/* Pipeline Progress */}
            {(isProcessing || currentPhase > 0) && (
              <div className="pipeline-progress">
                {PHASES.map((phase) => {
                  let status = ''
                  if (phase.id < currentPhase) status = 'complete'
                  else if (phase.id === currentPhase) status = isProcessing ? 'active' : 'complete'

                  return (
                    <motion.div
                      key={phase.id}
                      className={`pipeline-phase ${status}`}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: phase.id * 0.1 }}
                    >
                      <div className={`pipeline-phase-card ${status}`}>
                        <div className="phase-number">
                          {status === 'complete' ? <CheckCircle2 size={16} /> : phase.id}
                        </div>
                        <p className="phase-name">{phase.name}</p>
                        <p className="phase-time">{phase.description}</p>
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            )}

            {/* Loading Animation */}
            {isProcessing && (
              <div className="loading-spinner">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="bar" />
                ))}
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="diagnostic-result"
            style={{
              background: 'var(--accent-danger-dim)',
              border: '1px solid var(--border-danger)',
            }}
          >
            <AlertCircle size={24} color="var(--accent-danger)" />
            <div>
              <p style={{ fontWeight: 700, color: 'var(--accent-danger)' }}>Error</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{error}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Diagnostic Badge */}
            <div className={`diagnostic-result ${result.prediction?.toLowerCase() === 'malignant' ? 'malignant' : 'benign'}`}>
              <div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                  DIAGNOSIS
                </p>
                <p className="diagnostic-label">
                  {result.prediction?.toLowerCase() === 'malignant' ? '🔴' : '🟢'}{' '}
                  {result.prediction?.toUpperCase()}
                </p>
              </div>

              {/* Confidence Gauge */}
              <div className="confidence-gauge">
                <svg viewBox="0 0 120 120">
                  <circle className="confidence-gauge-bg" cx="60" cy="60" r="50" />
                  <circle
                    className="confidence-gauge-fill"
                    cx="60" cy="60" r="50"
                    style={{
                      stroke: result.prediction?.toLowerCase() === 'malignant'
                        ? 'var(--accent-danger)' : 'var(--accent-success)',
                      strokeDasharray: `${2 * Math.PI * 50}`,
                      strokeDashoffset: `${2 * Math.PI * 50 * (1 - (result.confidence || 0))}`,
                    }}
                  />
                </svg>
                <div className="confidence-gauge-text">
                  <span className="confidence-gauge-value">
                    {((result.confidence || 0) * 100).toFixed(1)}%
                  </span>
                  <span className="confidence-gauge-label">Confidence</span>
                </div>
              </div>

              {/* Trust Score */}
              <div style={{ textAlign: 'center' }}>
                {getTrustIcon(result.trust_score)}
                <p style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>Trust Score</p>
                <span className={`badge ${getTrustBadge(result.trust_score)}`}>
                  {result.trust_score}
                </span>
              </div>

              {/* Inference Time */}
              <div style={{ textAlign: 'center' }}>
                <Clock size={20} color="var(--accent-primary)" />
                <p style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>Inference</p>
                <p style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.9rem' }}>
                  {result.inference_time_ms}ms
                </p>
              </div>
            </div>

            {/* Visual Results Grid */}
            {result.visualizations && (
              <>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 'var(--space-xl) 0 var(--space-md)' }}>
                  📸 Visual Analysis
                </h3>
                <div className="results-grid">
                  {[
                    { key: 'original', label: 'Original Scan' },
                    { key: 'detection_overlay', label: 'YOLO Detection' },
                    { key: 'segmentation_mask', label: 'Segmentation Mask' },
                    { key: 'gradcam_overlay', label: 'Grad-CAM++ Heatmap' },
                  ].map(item => (
                    <motion.div
                      key={item.key}
                      className="result-image-card"
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.2 }}
                    >
                      {result.visualizations[item.key] ? (
                        <img
                          src={`data:image/png;base64,${result.visualizations[item.key]}`}
                          alt={item.label}
                        />
                      ) : (
                        <div style={{
                          aspectRatio: 1,
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: 'var(--text-secondary)',
                          fontSize: '0.85rem',
                          background: 'var(--bg-tertiary)',
                          padding: 'var(--space-md)',
                          textAlign: 'center',
                        }}>
                          <span style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>🔍</span>
                          {item.key === 'detection_overlay' && "No Tumor Bounding Box Detected"}
                          {item.key === 'segmentation_mask' && "No Tumor Region Segmented"}
                          {item.key === 'gradcam_overlay' && "No Heatmap Generated"}
                        </div>
                      )}
                      <div className="card-label">{item.label}</div>
                    </motion.div>
                  ))}
                </div>
              </>
            )}

            {/* Model Details */}
            {result.details && (
              <div className="stats-grid" style={{ marginTop: 'var(--space-xl)' }}>
                <div className="stat-card">
                  <p className="stat-card-label">ResNet50</p>
                  <p className="stat-card-value" style={{
                    color: result.details.resnet_prediction === 'Malignant' ? 'var(--accent-danger)' : 'var(--accent-success)'
                  }}>
                    {result.details.resnet_prediction}
                  </p>
                  <p style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                    {((result.details.resnet_confidence || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="stat-card">
                  <p className="stat-card-label">EfficientNetV2</p>
                  <p className="stat-card-value" style={{
                    color: result.details.efficient_prediction === 'Malignant' ? 'var(--accent-danger)' : 'var(--accent-success)'
                  }}>
                    {result.details.efficient_prediction}
                  </p>
                  <p style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                    {((result.details.efficient_confidence || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="stat-card">
                  <p className="stat-card-label">Tumor Detected</p>
                  <p className="stat-card-value" style={{ color: result.details.tumor_detected ? 'var(--accent-success)' : 'var(--text-tertiary)' }}>
                    {result.details.tumor_detected ? 'Yes' : 'No'}
                  </p>
                </div>
              </div>
            )}

            {/* Clinical Interpretation Card */}
            <div className="glass-card" style={{ marginTop: 'var(--space-xl)', padding: 'var(--space-lg)', borderLeft: `4px solid ${result.prediction?.toLowerCase() === 'malignant' ? 'var(--accent-danger)' : 'var(--accent-success)'}` }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 'var(--space-sm)', color: result.prediction?.toLowerCase() === 'malignant' ? 'var(--accent-danger)' : 'var(--accent-success)' }}>
                🏥 Clinical Interpretation & Guidance
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)', fontSize: '0.9rem', lineHeight: '1.5' }}>
                <div>
                  <p style={{ fontWeight: 600, margin: 0 }}>What does this result mean?</p>
                  <p style={{ color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                    {result.prediction?.toLowerCase() === 'malignant' 
                      ? "The AI models have detected characteristics highly suggestive of a malignant breast lesion (cancer). Malignant tumors typically exhibit irregular margins, shadowed boundaries, and micro-lobulations on ultrasound scans."
                      : result.details.tumor_detected
                        ? "A breast lesion/tumor was detected by the YOLO model, but the classification ensemble determined that it has benign characteristics (non-cancerous, such as a fibroadenoma or fluid-filled cyst). These lesions are generally harmless."
                        : "No suspicious breast tumors or masses were detected by the YOLO detection model and Attention U-Net. The scanned breast tissue appears normal and healthy."
                    }
                  </p>
                </div>

                <div>
                  <p style={{ fontWeight: 600, margin: 0 }}>Pipeline Analysis (Why this result occurred):</p>
                  <p style={{ color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                    {result.prediction?.toLowerCase() === 'malignant'
                      ? `YOLOv8-Seg isolated a suspect tumor region, which the Attention U-Net then mapped. EfficientNetV2 analyzed the shape and margins, predicting Malignant with ${((result.details.efficient_confidence || 0) * 100).toFixed(1)}% confidence.`
                      : result.details.tumor_detected
                        ? `YOLOv8-Seg successfully isolated a lesion, and the Attention U-Net mapped its boundaries. However, the classifier ensemble analyzed the shape and determined it lacks malignant features, predicting Benign.`
                        : "The YOLOv8-Seg detector found no tumor regions matching the confidence threshold, and the Attention U-Net confirmed there are no segmentable tumor boundaries, automatically defaulting the scan to a normal/benign status."
                    }
                  </p>
                </div>

                <div style={{ background: 'var(--bg-tertiary)', padding: 'var(--space-sm) var(--space-md)', borderRadius: 'var(--radius-sm)', marginTop: 'var(--space-xs)' }}>
                  <p style={{ fontWeight: 600, margin: 0, color: 'var(--text-primary)' }}>Recommended Next Steps:</p>
                  <p style={{ color: 'var(--text-secondary)', margin: '0.25rem 0 0', fontSize: '0.85rem' }}>
                    {result.trust_score === 'LOW'
                      ? "⚠️ WARNING: Low trust score detected due to model disagreement. Immediate clinical verification by a radiologist or oncologist is required to confirm the diagnosis."
                      : result.prediction?.toLowerCase() === 'malignant'
                        ? "🚨 URGENT: Referral for a core needle biopsy or fine-needle aspiration (FNA) is recommended to confirm malignancy and determine the histopathological profile."
                        : result.details.tumor_detected
                          ? "📅 Follow-up ultrasound in 6 to 12 months is recommended to monitor the size of the benign lesion for any changes, as per standard BIRADS-3 clinical guidelines."
                          : "✅ Routine screening and self-examinations as per standard age-group guidelines. No further diagnostic steps required at this time."
                    }
                  </p>
                </div>
              </div>
            </div>

            <button className="btn btn-outline" style={{ marginTop: 'var(--space-xl)' }} onClick={resetDiagnosis}>
              New Diagnosis
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
