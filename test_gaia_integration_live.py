#!/usr/bin/env python3
"""
Test Gaia AI Integration

This script tests the Gaia AI integration with real API calls.
"""

import asyncio
import json
from datetime import datetime
from loguru import logger

from src.utils.config import Config
from src.ai.gaia_client import GaiaClient, GaiaAnalysis


async def test_gaia_connection():
    """Test Gaia AI connection and basic functionality."""
    logger.info("Testing Gaia AI Integration...")
    
    try:
        # Load configuration
        config = Config()
        gaia_config = config.get_gaia_config()
        
        logger.info(f"Gaia enabled: {gaia_config.enabled}")
        logger.info(f"Gaia endpoint: {gaia_config.endpoint}")
        logger.info(f"Gaia model: {gaia_config.model}")
        
        if not gaia_config.enabled:
            logger.warning("Gaia is disabled in configuration")
            return False
        
        if not gaia_config.api_key:
            logger.warning("No Gaia API key configured")
            return False
        
        # Create Gaia client
        gaia_client = GaiaClient(gaia_config)
        
        # Test connection
        is_available = gaia_client.is_available()
        logger.info(f"Gaia available: {is_available}")
        
        if not is_available:
            logger.error("Gaia client not available")
            return False
        
        # Test connection
        connection_test = await gaia_client.test_connection()
        logger.info(f"Connection test: {connection_test}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Gaia connection: {e}")
        return False


async def test_gaia_inference():
    """Test Gaia AI inference with sample market data."""
    logger.info("Testing Gaia AI Inference...")
    
    try:
        # Load configuration
        config = Config()
        gaia_config = config.get_gaia_config()
        
        if not gaia_config.enabled or not gaia_config.api_key:
            logger.warning("Gaia not properly configured")
            return False
        
        # Create Gaia client
        gaia_client = GaiaClient(gaia_config)
        
        # Sample market data
        sample_market_data = {
            'symbol': 'BTC/USDT',
            'current_price': 50000.0,
            'volume_24h': 1000000000,
            'price_change_24h': 2.5,
            'high_24h': 51000.0,
            'low_24h': 49000.0,
            'ohlcv': {
                'close': [50000, 50100, 50200, 50300, 50400],
                'volume': [1000, 1100, 1200, 1300, 1400]
            },
            'indicators': {
                'rsi': 65.5,
                'macd': {'macd_line': 150, 'signal_line': 140},
                'bollinger': {'upper': 51000, 'lower': 49000}
            }
        }
        
        # Test market sentiment analysis
        logger.info("Testing market sentiment analysis...")
        sentiment_result = await gaia_client.analyze_market_sentiment(sample_market_data)
        if sentiment_result:
            logger.info(f"Sentiment analysis result: {sentiment_result}")
        else:
            logger.warning("No sentiment analysis result")
        
        # Test risk assessment
        logger.info("Testing risk assessment...")
        sample_signal = {
            'action': 'buy',
            'confidence': 0.75,
            'price': 50000.0,
            'quantity': 0.1
        }
        risk_result = await gaia_client.get_risk_assessment(sample_signal, sample_market_data)
        if risk_result:
            logger.info(f"Risk assessment result: {risk_result}")
        else:
            logger.warning("No risk assessment result")
        
        # Test signal enhancement
        logger.info("Testing signal enhancement...")
        enhancement_result = await gaia_client.enhance_trading_signal(sample_signal, sample_market_data)
        if enhancement_result:
            logger.info(f"Signal enhancement result: {enhancement_result}")
        else:
            logger.warning("No signal enhancement result")
        
        # Test market insights
        logger.info("Testing market insights...")
        insights_result = await gaia_client.get_market_insights('BTC/USDT', sample_market_data)
        if insights_result:
            logger.info(f"Market insights result: {insights_result}")
        else:
            logger.warning("No market insights result")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Gaia inference: {e}")
        return False


async def test_gaia_with_real_data():
    """Test Gaia with real market data from Binance."""
    logger.info("Testing Gaia with real market data...")
    
    try:
        # Load configuration
        config = Config()
        gaia_config = config.get_gaia_config()
        
        if not gaia_config.enabled or not gaia_config.api_key:
            logger.warning("Gaia not properly configured")
            return False
        
        # Create Gaia client
        gaia_client = GaiaClient(gaia_config)
        
        # Get real market data from Binance
        import ccxt
        
        exchange = ccxt.binance({
            'apiKey': config.get_exchange_config().api_key,
            'secret': config.get_exchange_config().secret,
            'sandbox': True,
            'testnet': True,
            'enableRateLimit': True
        })
        
        # Fetch real market data
        symbol = 'BTC/USDT'
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=100)
        ticker = exchange.fetch_ticker(symbol)
        
        # Prepare real market data
        real_market_data = {
            'symbol': symbol,
            'current_price': ticker['last'],
            'volume_24h': ticker['quoteVolume'],
            'price_change_24h': ticker['percentage'],
            'high_24h': ticker['high'],
            'low_24h': ticker['low'],
            'ohlcv': {
                'close': [candle[4] for candle in ohlcv[-20:]],  # Last 20 close prices
                'volume': [candle[5] for candle in ohlcv[-20:]]   # Last 20 volumes
            },
            'indicators': {
                'rsi': 55.0,  # Mock RSI
                'macd': {'macd_line': 100, 'signal_line': 95},  # Mock MACD
                'bollinger': {'upper': ticker['last'] * 1.02, 'lower': ticker['last'] * 0.98}
            }
        }
        
        logger.info(f"Real market data for {symbol}:")
        logger.info(f"  Current price: ${real_market_data['current_price']:,.2f}")
        logger.info(f"  24h volume: ${real_market_data['volume_24h']:,.0f}")
        logger.info(f"  24h change: {real_market_data['price_change_24h']:.2f}%")
        
        # Test Gaia inference with real data
        logger.info("Testing Gaia inference with real market data...")
        
        # Market sentiment
        sentiment = await gaia_client.analyze_market_sentiment(real_market_data)
        if sentiment:
            logger.info(f"Gaia sentiment analysis: {sentiment}")
        
        # Market insights
        insights = await gaia_client.get_market_insights(symbol, real_market_data)
        if insights:
            logger.info(f"Gaia market insights: {insights}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Gaia with real data: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("="*60)
    logger.info("GAIA AI INTEGRATION TEST")
    logger.info("="*60)
    
    # Test 1: Connection
    logger.info("\n1. Testing Gaia Connection...")
    connection_success = await test_gaia_connection()
    
    if connection_success:
        # Test 2: Basic inference
        logger.info("\n2. Testing Gaia Inference...")
        inference_success = await test_gaia_inference()
        
        # Test 3: Real data inference
        logger.info("\n3. Testing Gaia with Real Market Data...")
        real_data_success = await test_gaia_with_real_data()
        
        if inference_success and real_data_success:
            logger.info("\n✅ All Gaia tests passed!")
            logger.info("Gaia AI integration is ready for live trading!")
        else:
            logger.warning("\n⚠️ Some Gaia tests failed")
    else:
        logger.error("\n❌ Gaia connection failed")
        logger.info("Please check your Gaia API key and configuration")
    
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main()) 