#!/usr/bin/env python3
"""
Test script for Gaia LLM strategy with fixed connection.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.strategies.gaia_llm_strategy import GaiaLLMStrategy
import pandas as pd
import numpy as np

async def test_gaia_strategy():
    """Test the Gaia LLM strategy with mock data."""
    
    # Load config
    config = Config("config/config.yaml")
    strategy_config = config.get_strategy_config()
    
    # Create Gaia LLM strategy
    strategy = GaiaLLMStrategy(strategy_config.to_dict())
    
    # Create mock market data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    prices = np.random.randn(100).cumsum() + 50000  # BTC-like prices
    
    mock_ohlcv = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    market_data = {
        'ohlcv': mock_ohlcv,
        'timestamp': pd.Timestamp.now().timestamp(),
        'symbol': 'BTC/USDT'
    }
    
    print("Testing Gaia LLM Strategy...")
    print("=" * 50)
    
    try:
        # Generate signal
        signal = await strategy.generate_signal('BTC/USDT', market_data)
        
        if signal:
            print(f"✅ Gaia LLM Signal: {signal.signal_type}")
            print(f"   Confidence: {signal.confidence:.2f}")
            print(f"   Price: ${signal.price:,.2f}")
            print(f"   Strategy: {signal.strategy_name}")
            
            if hasattr(signal, 'metadata') and 'llm_response' in signal.metadata:
                print(f"   LLM Response: {signal.metadata['llm_response'][:200]}...")
        else:
            print("❌ No signal generated (LLM suggested HOLD)")
            
    except Exception as e:
        print(f"❌ Error testing Gaia LLM strategy: {e}")
        print("This might indicate a connection issue.")

if __name__ == "__main__":
    asyncio.run(test_gaia_strategy()) 