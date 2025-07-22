"""
Performance Tracking Module

Tracks trading performance metrics including P&L, Sharpe ratio, win rate, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

from ..utils.config import Config
from ..strategies.types import TradeSignal


class PerformanceTracker:
    """
    Comprehensive performance tracking system for the trading agent.
    
    Tracks:
    - Total P&L and percentage returns
    - Sharpe ratio and risk-adjusted returns
    - Win rate and average win/loss
    - Maximum drawdown
    - Trade frequency and duration
    - Competition metrics
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.performance_config = config.get_performance_config()
        self.competition_config = config.get_competition_config()
        
        # Performance data
        self.trades = []
        self.daily_returns = []
        self.balance_history = []
        self.peak_balance = 0.0
        self.initial_balance = 1000.0  # Default starting balance
        
        # Metrics tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_pnl_percentage = 0.0
        
        # Competition tracking
        self.competition_start_time = None
        self.competition_metrics = {}
        
        logger.info("Performance tracker initialized")
    
    async def start(self):
        """Start performance tracking."""
        self.competition_start_time = datetime.now()
        logger.info("Performance tracking started")
    
    async def stop(self):
        """Stop performance tracking."""
        logger.info("Performance tracking stopped")
    
    async def record_trade(self, signal: TradeSignal, order: Dict):
        """
        Record a completed trade for performance tracking.
        
        Args:
            signal: The trading signal that was executed
            order: The order information from the exchange
        """
        try:
            trade_data = {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'action': signal.action,
                'quantity': signal.quantity,
                'price': signal.price,
                'order_id': order.get('id', ''),
                'strategy': signal.strategy,
                'confidence': signal.confidence,
                'metadata': signal.metadata
            }
            
            # Calculate trade P&L if this is a sell order
            if signal.action == 'sell':
                pnl = await self._calculate_trade_pnl(signal, order)
                trade_data['pnl'] = pnl
                trade_data['pnl_percentage'] = (pnl / (signal.quantity * signal.price)) * 100
                
                # Update performance metrics
                self.total_pnl += pnl
                self.total_trades += 1
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            # Add to trades list
            self.trades.append(trade_data)
            
            # Update balance history
            await self._update_balance_history()
            
            logger.info(f"Trade recorded: {signal.action} {signal.symbol} "
                       f"@ {signal.price} (P&L: {trade_data.get('pnl', 0):.2f})")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    async def update_metrics(self):
        """Update performance metrics."""
        try:
            # Calculate daily returns
            await self._calculate_daily_returns()
            
            # Update competition metrics
            await self._update_competition_metrics()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _calculate_trade_pnl(self, signal: TradeSignal, order: Dict) -> float:
        """Calculate P&L for a completed trade."""
        try:
            # For now, we'll use a simplified P&L calculation
            # In a real implementation, you'd track entry and exit prices
            
            if signal.action == 'sell':
                # Assume we have the entry price from position tracking
                # This is a simplified calculation
                entry_price = signal.metadata.get('entry_price', signal.price * 0.99)
                pnl = (signal.price - entry_price) * signal.quantity
                return pnl
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating trade P&L: {e}")
            return 0.0
    
    async def _calculate_daily_returns(self):
        """Calculate daily returns for Sharpe ratio calculation."""
        try:
            if len(self.trades) == 0:
                return
            
            # Group trades by day
            trades_df = pd.DataFrame(self.trades)
            
            # Ensure 'pnl' column exists
            if 'pnl' not in trades_df.columns:
                trades_df['pnl'] = 0.0
                
            # Ensure timestamp is a datetime object before processing
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['date'] = trades_df['timestamp'].dt.date
            
            # Calculate daily P&L
            daily_pnl = trades_df.groupby('date')['pnl'].sum()
            
            # Calculate daily returns
            daily_returns = daily_pnl / self.initial_balance
            
            self.daily_returns = daily_returns.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating daily returns: {e}")
    
    async def _update_balance_history(self):
        """Update balance history for drawdown calculation."""
        try:
            current_balance = self.initial_balance + self.total_pnl
            self.balance_history.append({
                'timestamp': datetime.now(),
                'balance': current_balance,
                'pnl': self.total_pnl
            })
            
            # Update peak balance
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
                
        except Exception as e:
            logger.error(f"Error updating balance history: {e}")
    
    async def _update_competition_metrics(self):
        """Update competition-specific metrics."""
        try:
            if not self.competition_config.competition_mode:
                return
            
            # Calculate competition metrics
            self.competition_metrics = {
                'agent_id': self.competition_config.agent_id,
                'total_pnl': self.total_pnl,
                'total_pnl_percentage': self.total_pnl_percentage,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'max_drawdown': self._calculate_max_drawdown(),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor(),
                'avg_trade_duration': self._calculate_avg_trade_duration(),
                'competition_duration': (datetime.now() - self.competition_start_time).total_seconds() / 3600,  # hours
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating competition metrics: {e}")
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (risk-adjusted return)."""
        try:
            if len(self.daily_returns) < 2:
                return 0.0
            
            returns = np.array(self.daily_returns)
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252))
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(self.balance_history) < 2:
                return 0.0
            
            balances = [entry['balance'] for entry in self.balance_history]
            peak = balances[0]
            max_drawdown = 0.0
            
            for balance in balances:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage."""
        try:
            if self.total_trades == 0:
                return 0.0
            
            return (self.winning_trades / self.total_trades) * 100
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        try:
            if len(self.trades) == 0:
                return 0.0
            
            trades_df = pd.DataFrame(self.trades)
            
            # Ensure 'pnl' column exists
            if 'pnl' not in trades_df.columns:
                return 0.0
                
            winning_trades = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            losing_trades = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            
            if losing_trades == 0:
                return float('inf') if winning_trades > 0 else 0.0
            
            return winning_trades / losing_trades
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in hours."""
        try:
            if len(self.trades) < 2:
                return 0.0
            
            # Calculate time between trades
            timestamps = [pd.to_datetime(trade['timestamp']) for trade in self.trades]
            durations = []
            
            for i in range(1, len(timestamps)):
                duration = timestamps[i] - timestamps[i-1]
                durations.append(duration / 3600)  # Convert to hours
            
            return np.mean(durations) if durations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating avg trade duration: {e}")
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            current_balance = self.initial_balance + self.total_pnl
            self.total_pnl_percentage = (self.total_pnl / self.initial_balance) * 100
            
            return {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self._calculate_win_rate(),
                'total_pnl': self.total_pnl,
                'total_pnl_percentage': self.total_pnl_percentage,
                'current_balance': current_balance,
                'peak_balance': self.peak_balance,
                'max_drawdown': self._calculate_max_drawdown(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'profit_factor': self._calculate_profit_factor(),
                'avg_trade_duration': self._calculate_avg_trade_duration(),
                'competition_metrics': self.competition_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def get_competition_report(self) -> Dict[str, Any]:
        """Get competition-specific performance report."""
        if not self.competition_config.competition_mode:
            return {}
        
        return {
            'agent_id': self.competition_config.agent_id,
            'competition_start': self.competition_start_time.isoformat() if self.competition_start_time else None,
            'performance_metrics': self.competition_metrics,
            'fair_start': self.competition_config.fair_start,
            'performance_reporting': self.competition_config.performance_reporting
        }
    
    def export_trades(self, filename: str = None) -> str:
        """Export trades to CSV file."""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trades_export_{timestamp}.csv"
            
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(filename, index=False)
            
            logger.info(f"Trades exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
            return "" 