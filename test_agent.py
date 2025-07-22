"""
Test script for the Crypto Trading AI Agent.

This script demonstrates the basic functionality of the trading agent
without requiring actual exchange connections.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import Config
from agent.trading_agent import TradingAgent
from strategies.strategy_manager import StrategyManager
from data.market_data import MarketDataProvider
from loguru import logger


class MockMarketDataProvider:
    """Mock market data provider for testing."""
    
    def __init__(self, config):
        self.config = config
        self.is_connected = True
    
    async def connect(self):
        logger.info("Mock market data provider connected")
    
    async def disconnect(self):
        logger.info("Mock market data provider disconnected")
    
    async def get_latest_data(self, symbol: str):
        """Generate mock market data."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate mock OHLCV data
        timestamps = pd.date_range(end=datetime.now(), periods=100, freq='h')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        
        returns = np.random.normal(0, 0.02, len(timestamps))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        ohlcv_data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000, 10000)
            
            ohlcv_data.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'ohlcv': df,
            'current_price': prices[-1],
            'volume_24h': np.random.uniform(1000000, 5000000),
            'price_change_24h': np.random.uniform(-10, 10),
            'high_24h': max(prices),
            'low_24h': min(prices),
            'available_balance': 10000
        }


async def test_trading_agent():
    """Test the trading agent with mock data."""
    logger.info("Starting trading agent test...")
    
    try:
        # Load configuration
        config = Config("config/config.example.yaml")
        
        # Create mock market data provider
        mock_data_provider = MockMarketDataProvider(config)
        
        # Create strategy manager
        strategy_manager = StrategyManager(config)
        
        # Test strategy signal generation
        logger.info("Testing strategy signal generation...")
        
        symbols = config.get_trading_config().symbols[:2]  # Test with first 2 symbols
        
        for symbol in symbols:
            logger.info(f"Testing {symbol}...")
            
            # Get mock market data
            market_data = await mock_data_provider.get_latest_data(symbol)
            
            # Generate signals from all strategies
            signals = await strategy_manager.generate_signals(symbol, market_data)
            
            logger.info(f"Generated {len(signals)} signals for {symbol}")
            
            for signal in signals:
                logger.info(f"  - {signal.strategy_name}: {signal.action} "
                          f"(confidence: {signal.confidence:.2f})")
            
            # Combine signals
            if signals:
                combined_signal = await strategy_manager.combine_signals(signals)
                if combined_signal:
                    logger.info(f"Combined signal: {combined_signal.action} "
                              f"(confidence: {combined_signal.confidence:.2f})")
        
        # Test performance metrics
        logger.info("Testing performance metrics...")
        performance = await strategy_manager.get_strategy_performance()
        
        for strategy_name, metrics in performance.items():
            logger.info(f"{strategy_name}: {metrics}")
        
        logger.info("Trading agent test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


async def test_gaia_integration():
    """Test Gaia integration (simulated)."""
    logger.info("Testing Gaia integration...")
    
    try:
        config = Config("config/config.example.yaml")
        
        # Create Gaia client
        from deployment.gaia_client import GaiaClient
        gaia_client = GaiaClient(config)
        
        # Test connection status
        status = gaia_client.get_connection_status()
        logger.info(f"Gaia connection status: {status}")
        
        # Simulate AI analysis
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'current_price': 50000,
            'volume_24h': 2000000,
            'price_change_24h': 2.5
        }
        
        mock_strategy_signals = [
            {
                'strategy': 'rsi',
                'action': 'buy',
                'confidence': 0.7
            },
            {
                'strategy': 'macd',
                'action': 'buy',
                'confidence': 0.6
            }
        ]
        
        # This would normally call the actual Gaia API
        logger.info("Gaia integration test completed (simulated)")
        
    except Exception as e:
        logger.error(f"Gaia integration test failed: {e}")


async def main():
    """Main test function."""
    logger.info("=== Crypto Trading AI Agent Test ===")
    
    # Test basic functionality
    await test_trading_agent()
    
    # Test Gaia integration
    await test_gaia_integration()
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main()) 