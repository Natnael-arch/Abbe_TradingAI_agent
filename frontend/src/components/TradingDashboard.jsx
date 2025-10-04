import React, { useState, useEffect } from 'react'
import StatusPanel from './StatusPanel'
import ControlPanel from './ControlPanel'
import ConfigSelector from './ConfigSelector'

const API_BASE_URL = 'https://0d70a4df28e3.ngrok-free.app'

const TradingDashboard = () => {
  const [status, setStatus] = useState({
    status: 'stopped',
    signal: 'hold',
    pnl_percentage: 0,
    sentiment: 'neutral',
    last_update: 0,
    active_positions: 0,
    total_trades: 0,
    uptime: 0
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedConfig, setSelectedConfig] = useState('sandbox')

  // Fetch status from API
  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setStatus(data)
      setError(null)
    } catch (err) {
      console.error('Error fetching status:', err)
      setError('Failed to fetch status from server')
    }
  }

  // Control bot functions
  const startBot = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/start-bot`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      if (data.success) {
        // Immediately fetch updated status
        await fetchStatus()
      } else {
        setError(data.message)
      }
    } catch (err) {
      console.error('Error starting bot:', err)
      setError('Failed to start bot')
    } finally {
      setLoading(false)
    }
  }

  const stopBot = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/stop-bot`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      if (data.success) {
        // Immediately fetch updated status
        await fetchStatus()
      } else {
        setError(data.message)
      }
    } catch (err) {
      console.error('Error stopping bot:', err)
      setError('Failed to stop bot')
    } finally {
      setLoading(false)
    }
  }

  // Poll for status updates every 5 seconds
  useEffect(() => {
    // Initial fetch
    fetchStatus()
    
    // Set up polling
    const interval = setInterval(fetchStatus, 5000)
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="max-w-6xl mx-auto">
      {/* Configuration Selector */}
      <ConfigSelector 
        selectedConfig={selectedConfig}
        onConfigChange={setSelectedConfig}
      />

      {/* Error Message */}
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Error
              </h3>
              <div className="mt-2 text-sm text-red-700">
                {error}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control Panel */}
        <div className="lg:col-span-1">
          <ControlPanel
            status={status.status}
            onStart={startBot}
            onStop={stopBot}
            loading={loading}
          />
        </div>

        {/* Status Panel */}
        <div className="lg:col-span-2">
          <StatusPanel status={status} />
        </div>
      </div>
    </div>
  )
}

export default TradingDashboard
