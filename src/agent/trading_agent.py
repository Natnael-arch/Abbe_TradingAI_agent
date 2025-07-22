"""
Main Trading Agent Class

This is the core trading agent that orchestrates data collection, strategy execution,
risk management, and performance tracking.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_data import MarketDataProvider
from src.strategies.strategy_manager import StrategyManager
from src.strategies.types import TradeSignal
from src.risk.risk_manager import RiskManager
from src.performance.performance_tracker import PerformanceTracker
from src.utils.config import Config
from src.strategies.base_strategy import BaseStrategy
import pandas as pd
import traceback


def safe_float(val, context=''):
    try:
        return float(val)
    except Exception as e:
        logger.error(f"safe_float error in {context}: value={val} type={type(val)} error={e}\n{traceback.format_exc()}")
        raise


class TradingAgent:
    """
    Autonomous trading agent that manages the complete trading lifecycle.
    
    This agent:
    - Collects real-time market data
    - Executes trading strategies
    - Manages risk and position sizing
    - Tracks performance metrics
    - Operates continuously without human intervention
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.is_running = False
        self.last_update = 0
        
        # Initialize components
        self.market_data = MarketDataProvider(config)
        self.strategy_manager = StrategyManager(config)
        self.risk_manager = RiskManager(config)
        self.performance_tracker = PerformanceTracker(config)
        
        # State tracking
        self.active_positions = {}
        self.pending_orders = {}
        self.trade_history = []
        
        logger.info("Trading agent initialized successfully")
    
    async def start(self):
        """Start the trading agent."""
        logger.info("Starting trading agent...")
        self.is_running = True
        
        try:
            # Initialize market data connection
            await self.market_data.connect()
            
            # Start performance tracking
            await self.performance_tracker.start()
            
            # Main trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error in trading agent: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the trading agent."""
        logger.info("Stopping trading agent...")
        self.is_running = False
        
        # Close connections
        await self.market_data.disconnect()
        await self.performance_tracker.stop()
        
        logger.info("Trading agent stopped")
    
    async def _trading_loop(self):
        """Main trading loop that runs continuously."""
        update_interval = self.config.data.update_interval
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time for an update
                if current_time - self.last_update >= update_interval:
                    await self._process_market_update()
                    self.last_update = current_time
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_market_update(self):
        """Process a market update and execute trading decisions."""
        try:
            # Get latest market data for all symbols
            symbols = self.config.trading.symbols
            market_data = {}
            
            for symbol in symbols:
                data = await self.market_data.get_latest_data(symbol)
                if data is not None:
                    market_data[symbol] = data
            
            if not market_data:
                logger.warning("No market data available")
                return
            
            # Generate trading signals for each symbol
            signals = []
            for symbol, data in market_data.items():
                signal = await self._generate_signal(symbol, data)
                if signal:
                    signals.append(signal)
            
            # Execute trading decisions
            await self._execute_signals(signals)
            
            # Update performance metrics
            await self.performance_tracker.update_metrics()
            
        except Exception as e:
            logger.error(f"Error processing market update: {e}")
    
    async def _generate_signal(self, symbol: str, market_data: Dict) -> Optional[TradeSignal]:
        """Generate a trading signal for a given symbol."""
        try:
            # Calculate indicators and add to market_data for Gaia
            from src.strategies.hybrid_strategy import HybridStrategy
            base_strategy = HybridStrategy({'name': 'hybrid'})
            ohlcv = market_data.get('ohlcv')
            logger.info(f"Reached OHLCV cleaning for {symbol}, ohlcv type: {type(ohlcv)}")
            # Comprehensive data validation and cleaning
            logger.debug(f"Full market_data for {symbol}: {market_data}")
            # Clean ohlcv DataFrame
            if isinstance(ohlcv, pd.DataFrame):
                logger.debug(f"Before final cleaning, ohlcv['close'] values: {ohlcv['close'].tolist()}")
                # Convert all columns to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in ohlcv.columns:
                        ohlcv[col] = pd.to_numeric(ohlcv[col], errors='coerce')
                ohlcv = ohlcv.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                logger.debug(f"After cleaning, ohlcv['close'] values: {ohlcv['close'].tolist()}")
                close_prices = ohlcv['close'].values
                import numpy as np
                close_prices = np.array([v for v in close_prices if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isnull(v)])
                logger.debug(f"close_prices used for indicators: {close_prices}")
                market_data['ohlcv'] = ohlcv

            # Clean any other numeric fields in market_data
            for key, value in list(market_data.items()):
                if key not in ['ohlcv', 'symbol', 'timestamp']:
                    try:
                        if isinstance(value, str):
                            market_data[key] = float(value)
                        elif isinstance(value, (int, float)):
                            continue
                        elif value is not None:
                            # Try to convert arrays or lists to float if possible
                            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, dict)):
                                market_data[key] = [float(v) for v in value]
                    except Exception as e:
                        logger.error(f"Could not convert market_data['{key}'] to float: {value} (type: {type(value)}) - {e}")
                        market_data[key] = None
            if isinstance(ohlcv, pd.DataFrame) and len(ohlcv) >= 26:
                # Ensure 'close' column is numeric and drop NaNs
                ohlcv['close'] = pd.to_numeric(ohlcv['close'], errors='coerce')
                ohlcv = ohlcv.dropna(subset=['close'])
                close_prices = ohlcv['close'].values
                # Filter out non-numeric values
                import numpy as np
                close_prices = np.array([v for v in close_prices if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isnull(v)])
                # RSI
                if len(close_prices) >= 14:
                    rsi_val = base_strategy._calculate_rsi(close_prices, 14)[-1]
                    market_data['rsi'] = safe_float(rsi_val, context='RSI') if rsi_val is not None else None
                else:
                    logger.info(f"[GaiaLLM] Not enough data for RSI: {len(close_prices)} rows")
                # MACD
                if len(close_prices) >= 26:
                    macd = base_strategy._calculate_macd(close_prices, 12, 26, 9)
                    macd_line = macd['macd_line']
                    signal_line = macd['signal_line']
                    histogram = macd['histogram']
                    market_data['macd'] = safe_float(macd_line[-1], context='MACD') if len(macd_line) > 0 else None
                else:
                    logger.info(f"[GaiaLLM] Not enough data for MACD: {len(close_prices)} rows")
                # Bollinger Bands
                if len(close_prices) >= 20:
                    upper, middle, lower = base_strategy._calculate_bollinger(closes=close_prices, period=20, std_dev=2)
                    market_data['bb_upper'] = safe_float(upper[-1], context='BB_upper') if len(upper) > 0 and upper[-1] is not None else None
                    market_data['bb_lower'] = safe_float(lower[-1], context='BB_lower') if len(lower) > 0 and lower[-1] is not None else None
                else:
                    logger.info(f"[GaiaLLM] Not enough data for Bollinger Bands: {len(close_prices)} rows")
            else:
                logger.info(f"[GaiaLLM] ohlcv is not a DataFrame or too short: type={type(ohlcv)}, len={len(ohlcv) if ohlcv is not None else 'None'}")
            # Get strategy signals
            strategy_signals = await self.strategy_manager.generate_signals(symbol, market_data)
            if not strategy_signals:
                return None
            # Combine signals and get final decision
            final_signal = await self.strategy_manager.combine_signals(strategy_signals)
            if final_signal.action == 'hold':
                return None
            # Apply risk management
            risk_adjusted_signal = await self.risk_manager.apply_risk_management(
                symbol, final_signal, market_data
            )
            if risk_adjusted_signal:
                return risk_adjusted_signal
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
        return None
    
    async def _execute_signals(self, signals: List[TradeSignal]):
        """Execute trading signals."""
        for signal in signals:
            try:
                # Check if we can execute this trade
                if await self.risk_manager.can_execute_trade(signal):
                    # Execute the trade
                    order = await self.market_data.execute_trade(signal)
                    
                    if order:
                        # Update position tracking
                        await self._update_position(signal, order)
                        
                        # Log the trade
                        logger.info(f"Executed {signal.action} order for {signal.symbol}: "
                                  f"{signal.quantity} @ {signal.price}")
                        
                        # Update performance tracking
                        await self.performance_tracker.record_trade(signal, order)
                
            except Exception as e:
                logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    async def _update_position(self, signal: TradeSignal, order: Dict):
        """Update position tracking after a trade."""
        symbol = signal.symbol
        
        if signal.action == 'buy':
            if symbol not in self.active_positions:
                self.active_positions[symbol] = {
                    'quantity': signal.quantity,
                    'avg_price': signal.price,
                    'timestamp': signal.timestamp
                }
            else:
                # Update existing position
                current = self.active_positions[symbol]
                total_quantity = current['quantity'] + signal.quantity
                total_cost = (current['quantity'] * current['avg_price'] + 
                            signal.quantity * signal.price)
                current['avg_price'] = total_cost / total_quantity
                current['quantity'] = total_quantity
        
        elif signal.action == 'sell':
            if symbol in self.active_positions:
                # Calculate P&L
                position = self.active_positions[symbol]
                pnl = (signal.price - position['avg_price']) * signal.quantity
                
                # Update position
                remaining_quantity = position['quantity'] - signal.quantity
                if remaining_quantity <= 0:
                    del self.active_positions[symbol]
                else:
                    position['quantity'] = remaining_quantity
                
                logger.info(f"Position P&L for {symbol}: {pnl:.2f} USDT")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'is_running': self.is_running,
            'active_positions': len(self.active_positions),
            'pending_orders': len(self.pending_orders),
            'total_trades': len(self.trade_history),
            'last_update': self.last_update,
            'performance': self.performance_tracker.get_summary()
        }
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return self.active_positions.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_tracker.get_summary() 