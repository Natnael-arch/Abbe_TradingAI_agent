import React from 'react'

const ControlPanel = ({ status, onStart, onStop, loading }) => {
  const isRunning = status === 'running'
  const isStarting = status === 'starting'
  const isStopping = status === 'stopping'

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Bot Control
      </h2>
      
      <div className="space-y-4">
        {/* Status Indicator */}
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${
            isRunning ? 'bg-green-500' : 
            isStarting || isStopping ? 'bg-yellow-500 animate-pulse' : 
            'bg-gray-400'
          }`}></div>
          <span className="text-sm font-medium text-gray-700">
            Status: <span className="capitalize">{status}</span>
          </span>
        </div>

        {/* Control Buttons */}
        <div className="space-y-3">
          <button
            onClick={onStart}
            disabled={isRunning || isStarting || loading}
            className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${
              isRunning || isStarting || loading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {loading && isStarting ? (
              <div className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Starting...
              </div>
            ) : (
              'Start Bot'
            )}
          </button>

          <button
            onClick={onStop}
            disabled={!isRunning || isStopping || loading}
            className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${
              !isRunning || isStopping || loading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 text-white'
            }`}
          >
            {loading && isStopping ? (
              <div className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Stopping...
              </div>
            ) : (
              'Stop Bot'
            )}
          </button>
        </div>

        {/* Quick Stats */}
        <div className="pt-4 border-t border-gray-200">
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            Quick Stats
          </h3>
          <div className="text-xs text-gray-500 space-y-1">
            <div>Last Update: {new Date(status.last_update * 1000).toLocaleTimeString()}</div>
            {status.uptime > 0 && (
              <div>Uptime: {Math.floor(status.uptime / 60)}m {Math.floor(status.uptime % 60)}s</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ControlPanel









