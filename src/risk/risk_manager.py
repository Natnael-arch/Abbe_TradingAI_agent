"""
Risk Management Module

This module handles all risk management aspects of the trading agent including:
- Position sizing
- Stop-loss management
- Drawdown protection
- Daily trade limits
- Risk metrics calculation
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from ..utils.config import Config
from ..strategies.types import TradeSignal


@dataclass
class RiskMetrics:
    """Risk metrics for a trading session."""
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    trades_today: int
    max_daily_trades: int


class RiskManager:
    """
    Comprehensive risk management system for the trading agent.
    
    This class provides:
    - Position sizing based on risk parameters
    - Stop-loss and take-profit management
    - Drawdown protection
    - Daily trade limits
    - Real-time risk monitoring
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.risk_config = config.get_risk_config()
        
        # Risk state tracking
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_today = 0
        self.last_trade_date = None
        
        # Position tracking
        self.active_positions = {}
        self.stop_losses = {}
        self.take_profits = {}
        
        logger.info("Risk manager initialized successfully")
    
    async def apply_risk_management(self, symbol: str, signal: TradeSignal, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """
        Apply risk management rules to a trading signal.
        Now uses dynamic position sizing based on confidence.
        """
        try:
            # First, check if the trade can be executed at all
            if not await self.can_execute_trade(signal):
                return None
            
            # Calculate the quantity using our position sizing logic
            adjusted_quantity = await self._calculate_position_size(signal)
            logger.debug(f"Adjusted quantity for {symbol}: {adjusted_quantity}")

            if adjusted_quantity <= 0:
                logger.warning(f"Position size too small, rejecting signal: {signal.action}")
                return None
            
            # Apply stop-loss and take-profit
            risk_adjusted_signal = await self._apply_stop_loss_take_profit(signal, adjusted_quantity)
            logger.debug(f"Risk-adjusted signal: {risk_adjusted_signal}")
            
            return risk_adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying risk management: {e}")
            return None
    
    async def can_execute_trade(self, signal: TradeSignal) -> bool:
        """
        Check if a trade can be executed based on risk parameters.
        
        Args:
            signal: Trading signal to check
            
        Returns:
            True if trade can be executed, False otherwise
        """
        try:
            # Check daily trade limits
            if self.trades_today >= self.risk_config.max_daily_trades:
                logger.warning(f"Daily trade limit reached: {self.trades_today}")
                return False
            
            # Check drawdown limits
            if self.current_drawdown >= self.risk_config.max_drawdown:
                logger.warning(f"Maximum drawdown reached: {self.current_drawdown:.2%}")
                return False
            
            # Check position limits
            if not await self._check_position_limits(signal):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade execution: {e}")
            return False
    
    async def _check_basic_limits(self, signal: TradeSignal) -> bool:
        """Check basic risk limits."""
        # Check confidence threshold
        if signal.confidence < self.risk_config.min_confidence:
            logger.debug(f"Signal confidence too low: {signal.confidence:.2f}")
            return False
        
        # Check minimum order size
        min_order_value = self.risk_config.min_order_size
        order_value = signal.price * signal.quantity
        if order_value < min_order_value:
            logger.debug(f"Order value too small: {order_value:.2f}")
            return False
        
        return True
    
    async def _calculate_position_size(self, signal: TradeSignal) -> float:
        """
        Calculate position size based on risk parameters and signal confidence.
        """
        # Dynamic position sizing based on confidence
        confidence = getattr(signal, 'confidence', 0.5)
        # Fix: Get position_size from the correct config (trading_config)
        base_position_size_pct = self.config.get_trading_config().position_size

        if confidence >= 0.85:
            position_pct = base_position_size_pct * 1.0  # Use 100% of allowed size
        elif confidence >= 0.7:
            position_pct = base_position_size_pct * 0.75 # Use 75% of allowed size
        elif confidence >= 0.6:
            position_pct = base_position_size_pct * 0.5  # Use 50% of allowed size
        else:
            position_pct = base_position_size_pct * 0.25 # Use 25% of allowed size

        # For backtesting, we assume a fixed balance for sizing calculations
        # In live trading, this would be the actual available balance
        available_balance = 10000  # Assuming backtest balance for sizing

        position_size_usd = available_balance * position_pct
        quantity = position_size_usd / signal.price if signal.price > 0 else 0.0
        return quantity
    
    async def _apply_stop_loss_take_profit(self, signal, quantity: float):
        """Apply stop-loss and take-profit to the signal."""
        try:
            # Handle both StrategySignal and TradeSignal
            if hasattr(signal, 'strategy_name'):
                # StrategySignal
                strategy_name = signal.strategy_name
                action = signal.action
            else:
                # TradeSignal
                strategy_name = signal.strategy
                action = signal.action
            
            # Create risk-adjusted signal
            risk_signal = TradeSignal(
                symbol=signal.symbol,
                action=action,
                confidence=signal.confidence,
                price=signal.price,
                quantity=quantity,
                timestamp=signal.timestamp,
                strategy=strategy_name,
                metadata={
                    **signal.metadata,
                    'stop_loss': signal.price * (1 - self.risk_config.stop_loss_pct / 100),
                    'take_profit': signal.price * (1 + self.risk_config.take_profit_pct / 100),
                    'risk_adjusted': True
                }
            )
            
            return risk_signal
            
        except Exception as e:
            logger.error(f"Error applying stop-loss/take-profit: {e}")
            return signal
    
    async def _check_position_limits(self, signal: TradeSignal) -> bool:
        """Check position limits for the symbol."""
        try:
            symbol = signal.symbol
            
            # Check maximum position size
            if symbol in self.active_positions:
                current_position = self.active_positions[symbol]
                new_total = current_position['quantity'] + signal.quantity
                
                max_position_size = self.risk_config.max_position_size
                if new_total > max_position_size:
                    logger.warning(f"Position size limit exceeded: {new_total}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False
    
    async def _update_risk_metrics(self, signal: TradeSignal):
        """Update risk metrics after a signal."""
        try:
            # Update trade count
            current_date = time.strftime('%Y-%m-%d')
            if self.last_trade_date != current_date:
                self.trades_today = 0
                self.last_trade_date = current_date
            
            self.trades_today += 1
            
            # Update P&L (simplified calculation)
            if signal.action == 'sell':
                # Calculate P&L for sell orders
                if signal.symbol in self.active_positions:
                    position = self.active_positions[signal.symbol]
                    pnl = (signal.price - position['avg_price']) * signal.quantity
                    self.daily_pnl += pnl
                    self.total_pnl += pnl
            
            # Update drawdown
            if self.total_pnl < 0:
                self.current_drawdown = abs(self.total_pnl)
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return RiskMetrics(
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl,
            win_rate=self._calculate_win_rate(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            trades_today=self.trades_today,
            max_daily_trades=self.risk_config.max_daily_trades
        )
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate (simplified)."""
        # This would be calculated from actual trade history
        return 0.65  # Placeholder
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)."""
        # This would be calculated from actual returns
        return 1.2  # Placeholder
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update position tracking."""
        if symbol not in self.active_positions:
            self.active_positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'timestamp': time.time()
            }
        else:
            # Update existing position
            current = self.active_positions[symbol]
            total_quantity = current['quantity'] + quantity
            total_cost = (current['quantity'] * current['avg_price'] + quantity * price)
            current['avg_price'] = total_cost / total_quantity
            current['quantity'] = total_quantity
    
    def close_position(self, symbol: str, quantity: float, price: float):
        """Close a position."""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            remaining_quantity = position['quantity'] - quantity
            
            if remaining_quantity <= 0:
                del self.active_positions[symbol]
            else:
                position['quantity'] = remaining_quantity 