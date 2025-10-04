import React from 'react'
import TradingDashboard from './components/TradingDashboard'
import './App.css'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Trading Agent Dashboard
          </h1>
          <p className="text-gray-600">
            Monitor and control your AI trading bot
          </p>
        </header>
        
        <TradingDashboard />
      </div>
    </div>
  )
}

export default App