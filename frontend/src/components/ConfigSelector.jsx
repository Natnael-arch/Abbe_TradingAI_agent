import React from 'react'

const ConfigSelector = ({ selectedConfig, onConfigChange }) => {
  const configs = [
    {
      id: 'sandbox',
      name: 'Bitcoin Trading',
      description: 'Conservative BTC/USDT trading',
      symbols: ['BTC/USDT'],
      timeframe: '5m',
      risk: 'Low'
    },
    {
      id: 'memecoin',
      name: 'Memecoin Trading',
      description: 'High volatility memecoin trading',
      symbols: ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'BOME', 'MYRO'],
      timeframe: '1m',
      risk: 'High'
    }
  ]

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Trading Configuration
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {configs.map((config) => (
          <div
            key={config.id}
            className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
              selectedConfig === config.id
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => onConfigChange(config.id)}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-medium text-gray-900">
                {config.name}
              </h3>
              <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                config.risk === 'High' 
                  ? 'bg-red-100 text-red-800' 
                  : 'bg-green-100 text-green-800'
              }`}>
                {config.risk} Risk
              </div>
            </div>
            
            <p className="text-sm text-gray-600 mb-3">
              {config.description}
            </p>
            
            <div className="space-y-2">
              <div className="flex items-center text-sm text-gray-500">
                <span className="font-medium mr-2">Symbols:</span>
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                  {config.symbols.length} pairs
                </span>
              </div>
              
              <div className="flex items-center text-sm text-gray-500">
                <span className="font-medium mr-2">Timeframe:</span>
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                  {config.timeframe}
                </span>
              </div>
            </div>
            
            {selectedConfig === config.id && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <div className="text-xs text-gray-500">
                  <div className="font-medium mb-1">Trading Pairs:</div>
                  <div className="flex flex-wrap gap-1">
                    {config.symbols.map((symbol, index) => (
                      <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                        {symbol}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
      
      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
        <div className="flex items-start">
          <svg className="w-5 h-5 text-yellow-600 mt-0.5 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div className="text-sm text-yellow-800">
            <div className="font-medium">Configuration Note:</div>
            <div>Switching configurations will restart the trading bot with new parameters. Make sure to stop the bot before switching.</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ConfigSelector









