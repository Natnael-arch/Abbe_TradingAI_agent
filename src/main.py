"""
Main entry point for the Crypto Trading AI Agent.

This module:
- Sets up logging and configuration
- Initializes the trading agent
- Handles graceful shutdown
- Integrates with Gaia for AI inference
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.utils.config import Config
from src.agent.trading_agent import TradingAgent


class TradingAgentRunner:
    """Main runner class for the trading agent."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = None
        self.agent = None
        self.is_running = False
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = self.config.get_logging_config()
        
        # Remove default logger
        logger.remove()
        
        # Add console logger
        logger.add(
            sys.stdout,
            level=logging_config.level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>"
        )
        
        # Add file logger
        log_file = logging_config.file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logger.add(
            log_file,
            level=logging_config.level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=logging_config.max_size,
            retention=logging_config.backup_count,
            compression="zip"
        )
        
        logger.info("Logging setup complete")
    
    async def initialize(self):
        """Initialize the trading agent."""
        try:
            # Load configuration
            logger.info("Loading configuration...")
            self.config = Config(self.config_path)
            
            # Validate configuration
            self.config.validate()
            logger.info("Configuration validated successfully")
            
            # Setup logging
            self._setup_logging()
            
            # Initialize trading agent
            logger.info("Initializing trading agent...")
            self.agent = TradingAgent(self.config)
            
            # Log agent status
            logger.info("Trading agent initialized successfully")
            logger.info(f"Trading symbols: {self.config.get_trading_config().symbols}")
            logger.info(f"Strategy: {self.config.get_strategy_config().name}")
            logger.info(f"Risk management: Max drawdown {self.config.get_risk_config().max_drawdown}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading agent: {e}")
            return False
    
    async def start(self):
        """Start the trading agent."""
        if not await self.initialize():
            logger.error("Failed to initialize, exiting...")
            return
        
        self.is_running = True
        logger.info("Starting trading agent for a 30-minute live trading session...")
        
        try:
            # Start the agent and run for 30 minutes
            await asyncio.wait_for(self.agent.start(), timeout=30 * 60)
            
        except asyncio.TimeoutError:
            logger.info("30-minute trading session finished.")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in trading agent: {e}")
        finally:
            await self.shutdown()
            
            # Print final performance report
            if self.agent:
                summary = self.agent.get_performance_summary()
                logger.info("\n\n--- FINAL PERFORMANCE REPORT ---")
                for key, value in summary.items():
                    logger.info(f"{key}: {value}")
                logger.info("--------------------------------\n")

    
    async def shutdown(self):
        """Shutdown the trading agent gracefully."""
        logger.info("Shutting down trading agent...")
        
        if self.agent:
            await self.agent.stop()
        
        self.is_running = False
        logger.info("Trading agent shutdown complete")
    
    def get_status(self) -> dict:
        """Get current status of the trading agent."""
        if not self.agent:
            return {"status": "not_initialized"}
        
        return {
            "status": "running" if self.is_running else "stopped",
            "agent_status": self.agent.get_status(),
            "positions": self.agent.get_positions(),
            "performance": self.agent.get_performance_summary()
        }


async def main():
    """Main entry point."""
    # Check if config file exists
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please copy config/config.example.yaml to config/config.yaml and configure your settings")
        return
    
    # Create and run the trading agent
    runner = TradingAgentRunner(config_path)
    
    try:
        # Run the agent for 30 minutes
        await asyncio.wait_for(runner.start(), timeout=30 * 60)
    except asyncio.TimeoutError:
        logger.info("30-minute trading session finished. Shutting down...")
    finally:
        # Ensure graceful shutdown
        if runner.is_running:
            await runner.shutdown()
            
        # Print final performance summary
        if runner.agent:
            summary = runner.agent.get_performance_summary()
            print("\n=== Final Performance Summary ===")
            for k, v in summary.items():
                print(f"{k}: {v}")
            
            # Export trades
            csv_file = runner.agent.performance_tracker.export_trades()
            print(f"\nTrade history exported to: {csv_file}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 