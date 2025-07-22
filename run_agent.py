#!/usr/bin/env python3
"""
Simple runner for the Crypto Trading AI Agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.agent.trading_agent import TradingAgent
from loguru import logger


async def main():
    """Main entry point."""
    try:
        # Check if config file exists
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Please copy config/config.example.yaml to config/config.yaml and configure your settings")
            return
        
        # Load configuration
        logger.info("Loading configuration...")
        config = Config(config_path)
        
        # Validate configuration
        config.validate()
        logger.info("Configuration validated successfully")
        
        # Initialize trading agent
        logger.info("Initializing trading agent...")
        agent = TradingAgent(config)
        
        # Log agent status
        logger.info("Trading agent initialized successfully")
        logger.info(f"Trading symbols: {config.get_trading_config().symbols}")
        logger.info(f"Strategy: {config.get_strategy_config().name}")
        logger.info(f"Risk management: Max drawdown {config.get_risk_config().max_drawdown}")
        
        # Start the agent
        logger.info("Starting trading agent...")
        await agent.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running trading agent: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 