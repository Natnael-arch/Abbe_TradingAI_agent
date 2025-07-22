#!/usr/bin/env python3
"""
Quick test script for the competition strategy.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.strategies.competition_strategy import CompetitionStrategy
import pandas as pd
import numpy as np

async def test_competition_strategy():
    """Test the competition strategy with mock data."""
    
    # Load config
    config = Config("config/config.yaml")
    strategy_config = config.get_strategy_config()
    
    # Create competition strategy
    strategy = CompetitionStrategy(strategy_config.to_dict())
    
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
    
    print("Testing Competition Strategy...")
    print("=" * 50)
    
    # Generate signals for multiple time periods
    for i in range(10):
        # Update market data with new prices
        new_price = 50000 + np.random.randn() * 1000
        mock_ohlcv.iloc[-1]['close'] = new_price
        mock_ohlcv.iloc[-1]['high'] = new_price * 1.01
        mock_ohlcv.iloc[-1]['low'] = new_price * 0.99
        
        # Generate signal
        signal = await strategy.generate_signal('BTC/USDT', market_data)
        
        if signal:
            print(f"Signal {i+1}: {signal.signal_type} | Confidence: {signal.confidence:.2f} | Price: ${signal.price:,.2f}")
            if hasattr(signal, 'metadata') and 'signal_analysis' in signal.metadata:
                analysis = signal.metadata['signal_analysis']
                print(f"  Strength: {analysis.get('signal_strength', 0):.2f}")
                print(f"  Tech Score: {analysis.get('tech_score', 0)}")
                print(f"  Momentum Score: {analysis.get('momentum_score', 0)}")
        else:
            print(f"Signal {i+1}: HOLD (no signal generated)")
        
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_competition_strategy()) 