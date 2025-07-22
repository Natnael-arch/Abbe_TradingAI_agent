#!/usr/bin/env python3
"""
Simple connection test for Binance testnet
"""

import ccxt
from loguru import logger


def test_binance_connection():
    """Test Binance testnet connection with actual API keys."""
    logger.info("Testing Binance Testnet connection...")
    
    try:
        # Create testnet exchange instance with your actual keys
        exchange = ccxt.binance({
            'apiKey': '8ILawj12QiElys2NEgccvJ6t8HSAWOK25YFx8DdZCzXEhIVur9ekwKh8CQcVBBNF',
            'secret': '6d7Fl0tFCPMuXKMA8KHH2zVjNustgUzcjH38pfsGfLJ1Ucs1y7UwHA2JpzhYmbe9',
            'sandbox': True,
            'testnet': True,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        # Test connection
        exchange.load_markets()
        logger.info("✅ Binance testnet connection successful!")
        
        # Test fetching data
        ticker = exchange.fetch_ticker('BTC/USDT')
        logger.info(f"✅ Current BTC price: ${ticker['last']:,.2f}")
        
        # Test account balance (will be 0 on testnet)
        balance = exchange.fetch_balance()
        logger.info(f"✅ Account balance: {balance['total']}")
        
        # Test fetching OHLCV data
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=10)
        logger.info(f"✅ Fetched {len(ohlcv)} OHLCV data points")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Binance testnet connection failed: {e}")
        return False


if __name__ == "__main__":
    success = test_binance_connection()
    if success:
        print("\n🎉 Connection successful! You can now run:")
        print("python run_agent.py")
        print("python backtest.py")
    else:
        print("\n❌ Connection failed. Please check your API keys.") 