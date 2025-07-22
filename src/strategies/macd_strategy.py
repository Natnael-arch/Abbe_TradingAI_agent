"""
MACD (Moving Average Convergence Divergence) Trading Strategy

This strategy uses MACD to identify trend changes and momentum.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from loguru import logger

from .base_strategy import BaseStrategy
from .strategy_manager import StrategySignal


class MACDStrategy(BaseStrategy):
    """
    MACD-based trading strategy.
    
    This strategy:
    - Calculates MACD line, signal line, and histogram
    - Generates buy signals on bullish crossovers
    - Generates sell signals on bearish crossovers
    - Adjusts confidence based on histogram strength
    """
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default MACD parameters."""
        return {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'min_confidence': 0.3,
            'max_confidence': 0.9,
            'histogram_threshold': 0.001  # Minimum histogram value for signal
        }
    
    async def generate_signal(self, symbol: str, market_data: Dict) -> Optional[StrategySignal]:
        """
        Generate MACD-based trading signal.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            StrategySignal if conditions are met, None otherwise
        """
        if not self._validate_market_data(market_data):
            return None
        
        try:
            ohlcv = market_data['ohlcv']
            current_price = market_data['current_price']
            
            # Calculate MACD
            macd_data = self._calculate_macd(
                ohlcv['close'].values,
                self.parameters['macd_fast'],
                self.parameters['macd_slow'],
                self.parameters['macd_signal']
            )
            
            if len(macd_data['macd_line']) < 2:
                return None
            
            # Get current and previous values
            current_macd = macd_data['macd_line'][-1]
            previous_macd = macd_data['macd_line'][-2]
            current_signal = macd_data['signal_line'][-1]
            previous_signal = macd_data['signal_line'][-2]
            current_histogram = macd_data['histogram'][-1]
            
            # Skip if values are NaN
            if (np.isnan(current_macd) or np.isnan(current_signal) or 
                np.isnan(previous_macd) or np.isnan(previous_signal)):
                return None
            
            # Determine action and confidence
            action, confidence = self._analyze_macd(
                current_macd, previous_macd,
                current_signal, previous_signal,
                current_histogram
            )
            
            if action == 'hold':
                return None
            
            # Calculate position size
            available_balance = market_data.get('available_balance', 1000)  # Default for testing
            quantity = self._calculate_position_size(available_balance, current_price, confidence)
            
            if quantity <= 0:
                return None
            
            # Create signal
            signal = StrategySignal(
                strategy_name='macd',
                action=action,
                confidence=confidence,
                price=current_price,
                quantity=quantity,
                metadata={
                    'symbol': symbol,
                    'macd_line': current_macd,
                    'signal_line': current_signal,
                    'histogram': current_histogram,
                    'timestamp': market_data.get('timestamp', 0),
                    'market_data': market_data
                }
            )
            
            # Record signal for performance tracking
            self._record_signal(signal)
            
            logger.info(f"MACD signal for {symbol}: {action} (MACD: {current_macd:.4f}, "
                       f"Signal: {current_signal:.4f}, Histogram: {current_histogram:.4f}, "
                       f"confidence: {confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating MACD signal for {symbol}: {e}")
            return None
    
    def _analyze_macd(self, current_macd: float, previous_macd: float,
                      current_signal: float, previous_signal: float,
                      current_histogram: float) -> tuple[str, float]:
        """
        Analyze MACD values and determine action and confidence.
        
        Args:
            current_macd: Current MACD line value
            previous_macd: Previous MACD line value
            current_signal: Current signal line value
            previous_signal: Previous signal line value
            current_histogram: Current histogram value
            
        Returns:
            Tuple of (action, confidence)
        """
        min_confidence = self.parameters['min_confidence']
        max_confidence = self.parameters['max_confidence']
        histogram_threshold = self.parameters['histogram_threshold']
        
        # Check for crossovers
        macd_crossover_up = (previous_macd <= previous_signal and current_macd > current_signal)
        macd_crossover_down = (previous_macd >= previous_signal and current_macd < current_signal)
        
        # Check histogram strength
        histogram_strong = abs(current_histogram) > histogram_threshold
        
        # Determine action
        if macd_crossover_up and histogram_strong:
            action = 'buy'
        elif macd_crossover_down and histogram_strong:
            action = 'sell'
        else:
            return 'hold', 0.0
        
        # Calculate confidence based on histogram strength and crossover quality
        base_confidence = min_confidence
        
        # Adjust confidence based on histogram strength
        histogram_strength = min(abs(current_histogram) / (histogram_threshold * 10), 1.0)
        confidence = base_confidence + (max_confidence - base_confidence) * histogram_strength
        
        # Ensure confidence is within bounds
        confidence = max(min_confidence, min(max_confidence, confidence))
        
        return action, confidence
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get MACD strategy performance metrics."""
        if not self.performance_history:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0.0,
                'avg_histogram': 0.0
            }
        
        # Calculate metrics
        total_signals = len(self.performance_history)
        buy_signals = sum(1 for s in self.performance_history if s['action'] == 'buy')
        sell_signals = sum(1 for s in self.performance_history if s['action'] == 'sell')
        avg_confidence = np.mean([s['confidence'] for s in self.performance_history])
        
        # Calculate average histogram for signals
        histogram_values = [s['metadata'].get('histogram', 0) for s in self.performance_history 
                          if 'metadata' in s and 'histogram' in s['metadata']]
        avg_histogram = np.mean(histogram_values) if histogram_values else 0.0
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'avg_histogram': avg_histogram,
            'strategy_name': 'macd'
        }
    
    def get_macd_analysis(self, market_data: Dict) -> Dict[str, Any]:
        """
        Get detailed MACD analysis for debugging and monitoring.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary with MACD analysis
        """
        if not self._validate_market_data(market_data):
            return {}
        
        try:
            ohlcv = market_data['ohlcv']
            macd_data = self._calculate_macd(
                ohlcv['close'].values,
                self.parameters['macd_fast'],
                self.parameters['macd_slow'],
                self.parameters['macd_signal']
            )
            
            if len(macd_data['macd_line']) < 2:
                return {}
            
            current_macd = macd_data['macd_line'][-1]
            current_signal = macd_data['signal_line'][-1]
            current_histogram = macd_data['histogram'][-1]
            
            action, confidence = self._analyze_macd(
                current_macd, macd_data['macd_line'][-2],
                current_signal, macd_data['signal_line'][-2],
                current_histogram
            )
            
            return {
                'macd_line': current_macd,
                'signal_line': current_signal,
                'histogram': current_histogram,
                'action': action,
                'confidence': confidence,
                'fast_period': self.parameters['macd_fast'],
                'slow_period': self.parameters['macd_slow'],
                'signal_period': self.parameters['macd_signal'],
                'histogram_threshold': self.parameters['histogram_threshold'],
                'trend': self._get_macd_trend(macd_data)
            }
            
        except Exception as e:
            logger.error(f"Error in MACD analysis: {e}")
            return {}
    
    def _get_macd_trend(self, macd_data: Dict[str, np.ndarray]) -> str:
        """Determine MACD trend based on recent values."""
        if len(macd_data['macd_line']) < 5:
            return 'unknown'
        
        recent_macd = macd_data['macd_line'][-5:]
        recent_histogram = macd_data['histogram'][-5:]
        
        # Check if MACD line is trending up
        if np.all(np.diff(recent_macd) > 0):
            return 'bullish'
        elif np.all(np.diff(recent_macd) < 0):
            return 'bearish'
        else:
            return 'sideways' 