#!/usr/bin/env python3
"""
Test script for the hybrid strategy.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategies.hybrid_strategy import HybridStrategy
from src.utils.config import StrategyConfig


def test_hybrid_strategy():
    """Test the hybrid strategy with mock data."""
    print("Testing Hybrid Strategy...")
    
    # Create strategy config
    config = StrategyConfig(
        name='hybrid',
        enabled=True,
        weight=1.0,
        parameters={
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'volume_sma_period': 20,
            'volume_threshold': 1.5,
            'min_indicators_agree': 2,
            'min_confidence': 0.4,
            'max_confidence': 0.95
        }
    )
    
    # Create strategy
    strategy = HybridStrategy(config.to_dict())
    
    # Generate mock market data
    timestamps = pd.date_range(end=datetime.now(), periods=100, freq='h')
    prices = []
    current_price = 50000  # BTC price
    
    for _ in range(100):
        change = np.random.normal(0, 0.02 / np.sqrt(24))
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    ohlcv_data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = price if i == 0 else prices[i-1]
        volume = np.random.uniform(1000, 10000)
        
        ohlcv_data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(ohlcv_data)
    df.set_index('timestamp', inplace=True)
    
    # Create market data
    market_data = {
        'symbol': 'BTC/USDT',
        'ohlcv': df,
        'current_price': current_price,
        'timestamp': datetime.now().isoformat()
    }
    
    # Test the strategy
    print(f"Testing with {len(df)} data points...")
    print(f"Current price: ${current_price:.2f}")
    
    # Test validation
    is_valid = strategy._validate_market_data(market_data)
    print(f"Market data validation: {is_valid}")
    
    if is_valid:
        # Test indicator calculation
        indicators = strategy._calculate_all_indicators(df)
        print(f"Indicators calculated: {list(indicators.keys())}")
        
        # Test signal generation
        import asyncio
        
        async def test_signal():
            signal = await strategy.generate_signal('BTC/USDT', market_data)
            if signal:
                print(f"Signal generated: {signal.action} with confidence {signal.confidence:.2f}")
            else:
                print("No signal generated")
        
        asyncio.run(test_signal())
    
    print("Test completed!")


if __name__ == "__main__":
    test_hybrid_strategy() 