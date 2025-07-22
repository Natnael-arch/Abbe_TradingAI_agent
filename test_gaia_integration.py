#!/usr/bin/env python3
"""
Test script for Gaia AI integration.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from src.ai.gaia_client import GaiaClient
from src.utils.config import GaiaConfig


async def test_gaia_integration():
    """Test Gaia AI integration with mock data."""
    print("Testing Gaia AI Integration...")
    
    # Create Gaia config (without API key for demo)
    config = GaiaConfig(
        enabled=True,
        api_key="",  # Add your API key here for real testing
        endpoint="https://api.gaianet.ai/v1",
        model="trading-ai-v1"
    )
    
    # Create Gaia client
    gaia_client = GaiaClient(config)
    
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
        'volume_24h': sum([row['volume'] for row in ohlcv_data]),
        'timestamp': datetime.now().isoformat(),
        'ticker': {
            'last': current_price,
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'high': max(prices),
            'low': min(prices),
            'volume': sum([row['volume'] for row in ohlcv_data])
        },
        'orderbook': {
            'bids': [[current_price * 0.999, 1.0] for _ in range(10)],
            'asks': [[current_price * 1.001, 1.0] for _ in range(10)]
        }
    }
    
    # Create mock trading signal
    signal_data = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'confidence': 0.75,
        'price': current_price,
        'quantity': 0.001,
        'timestamp': datetime.now().timestamp(),
        'strategy_name': 'hybrid',
        'metadata': {
            'rsi': 35.2,
            'macd_signal': 'bullish',
            'volume_confirmation': True
        }
    }
    
    print(f"Testing with {len(df)} data points...")
    print(f"Current price: ${current_price:.2f}")
    print(f"Signal: {signal_data['action']} with confidence {signal_data['confidence']:.2f}")
    
    # Test Gaia availability
    is_available = gaia_client.is_available()
    print(f"Gaia AI available: {is_available}")
    
    if is_available:
        # Test market insights
        print("\nTesting market insights...")
        insights = await gaia_client.get_market_insights('BTC/USDT', market_data)
        if insights:
            print(f"Market insights received: {insights}")
        else:
            print("No market insights available (expected without API key)")
        
        # Test signal enhancement
        print("\nTesting signal enhancement...")
        enhanced = await gaia_client.enhance_trading_signal(signal_data, market_data)
        if enhanced:
            print(f"Signal enhanced: {enhanced}")
        else:
            print("No signal enhancement available (expected without API key)")
        
        # Test risk assessment
        print("\nTesting risk assessment...")
        risk = await gaia_client.get_risk_assessment(signal_data, market_data)
        if risk:
            print(f"Risk assessment: {risk}")
        else:
            print("No risk assessment available (expected without API key)")
    
    print("\nGaia AI integration test completed!")
    print("\nTo enable real Gaia AI integration:")
    print("1. Get your API key from https://docs.gaianet.ai/")
    print("2. Add it to config/config.yaml")
    print("3. Restart the trading agent")


if __name__ == "__main__":
    asyncio.run(test_gaia_integration()) 