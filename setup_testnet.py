#!/usr/bin/env python3
"""
Testnet Setup Script

This script helps set up testnet API keys and test the connection.
"""

import asyncio
import ccxt
from loguru import logger
from src.utils.config import Config


async def test_binance_testnet():
    """Test Binance testnet connection."""
    logger.info("Testing Binance Testnet connection...")
    
    try:
        # Create testnet exchange instance
        exchange = ccxt.binance({
            'apiKey': 'YOUR_TESTNET_API_KEY',  # Replace with your testnet API key
            'secret': 'YOUR_TESTNET_SECRET_KEY',  # Replace with your testnet secret key
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
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Binance testnet connection failed: {e}")
        return False


async def test_coinbase_sandbox():
    """Test Coinbase Pro sandbox connection."""
    logger.info("Testing Coinbase Pro Sandbox connection...")
    
    try:
        # Create sandbox exchange instance
        exchange = ccxt.coinbasepro({
            'apiKey': 'YOUR_COINBASE_API_KEY',  # Replace with your Coinbase API key
            'secret': 'YOUR_COINBASE_SECRET_KEY',  # Replace with your Coinbase secret key
            'passphrase': 'YOUR_COINBASE_PASSPHRASE',  # Replace with your passphrase
            'sandbox': True,
            'enableRateLimit': True
        })
        
        # Test connection
        exchange.load_markets()
        logger.info("✅ Coinbase Pro sandbox connection successful!")
        
        # Test fetching data
        ticker = exchange.fetch_ticker('BTC-USD')
        logger.info(f"✅ Current BTC price: ${ticker['last']:,.2f}")
        
        # Test account balance
        balance = exchange.fetch_balance()
        logger.info(f"✅ Account balance: {balance['total']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Coinbase Pro sandbox connection failed: {e}")
        return False


def print_setup_instructions():
    """Print setup instructions for testnet APIs."""
    print("\n" + "="*60)
    print("TESTNET SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n📋 BINANCE TESTNET:")
    print("1. Go to: https://testnet.binance.vision/")
    print("2. Click 'Get API Key'")
    print("3. Log in with your Binance account")
    print("4. Copy the API Key and Secret Key")
    print("5. Update config/config.yaml with your keys")
    print("6. Run: python setup_testnet.py")
    
    print("\n📋 COINBASE PRO SANDBOX:")
    print("1. Go to: https://pro.coinbase.com/")
    print("2. Create account and enable API access")
    print("3. Generate API key with sandbox permissions")
    print("4. Copy API Key, Secret Key, and Passphrase")
    print("5. Update config/config_coinbase.yaml with your keys")
    print("6. Run: python setup_testnet.py --coinbase")
    
    print("\n💡 TIPS:")
    print("- Testnet accounts start with 0 balance")
    print("- You can request testnet funds from Binance")
    print("- All trades on testnet are virtual (no real money)")
    print("- Perfect for strategy testing and development")
    
    print("="*60)


async def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--coinbase':
        # Test Coinbase Pro sandbox
        success = await test_coinbase_sandbox()
    else:
        # Test Binance testnet
        success = await test_binance_testnet()
    
    if not success:
        print_setup_instructions()


if __name__ == "__main__":
    asyncio.run(main()) 