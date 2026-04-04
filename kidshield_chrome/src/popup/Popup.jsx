import { useState } from 'react'

function Popup() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  async function checkHealth() {
    setLoading(true)
    setStatus(null)
    try {
      const res = await fetch('http://127.0.0.1:8000/health')
      const data = await res.json()
      setStatus({ ok: true, message: data.status })
    } catch {
      setStatus({ ok: false, message: 'Server unreachable' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="popup-container">
      <h1 className="popup-title">KidShield AI</h1>
      <button className="health-btn" onClick={checkHealth} disabled={loading}>
        {loading ? 'Checking...' : 'Check Backend'}
      </button>
      {status && (
        <p className={`health-result ${status.ok ? 'ok' : 'error'}`}>
          {status.ok ? 'Yes it works' : 'it failed'} {status.message}
        </p>
      )}
    </div>
  )
}

export default Popup
