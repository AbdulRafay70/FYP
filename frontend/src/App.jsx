import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Layout/Sidebar'
import HomePage from './pages/HomePage'
import DiagnosticPage from './pages/DiagnosticPage'
import BenchmarkPage from './pages/BenchmarkPage'

function App() {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/diagnose" element={<DiagnosticPage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
