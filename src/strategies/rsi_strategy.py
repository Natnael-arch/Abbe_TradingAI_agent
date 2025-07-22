"""
RSI (Relative Strength Index) Trading Strategy

This strategy uses RSI to identify overbought and oversold conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from loguru import logger

from .base_strategy import BaseStrategy
from .strategy_manager import StrategySignal


class RSIStrategy(BaseStrategy):
    """
    RSI-based trading strategy.
    
    This strategy:
    - Calculates RSI for the given period
    - Generates buy signals when RSI is oversold (< 30)
    - Generates sell signals when RSI is overbought (> 70)
    - Adjusts confidence based on RSI extremity
    """
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default RSI parameters."""
        return {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_extreme_overbought': 80,
            'rsi_extreme_oversold': 20,
            'min_confidence': 0.3,
            'max_confidence': 0.9
        }
    
    async def generate_signal(self, symbol: str, market_data: Dict) -> Optional[StrategySignal]:
        """
        Generate RSI-based trading signal.
        
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
            
            # Calculate RSI
            rsi_values = self._calculate_rsi(ohlcv['close'].values, self.parameters['rsi_period'])
            
            if len(rsi_values) == 0:
                return None
            
            current_rsi = rsi_values[-1]
            
            # Skip if RSI is not available (NaN)
            if np.isnan(current_rsi):
                return None
            
            # Determine action and confidence
            action, confidence = self._analyze_rsi(current_rsi)
            
            if action == 'hold':
                return None
            
            # Calculate position size
            available_balance = market_data.get('available_balance', 1000)  # Default for testing
            quantity = self._calculate_position_size(available_balance, current_price, confidence)
            
            if quantity <= 0:
                return None
            
            # Create signal
            signal = StrategySignal(
                strategy_name='rsi',
                action=action,
                confidence=confidence,
                price=current_price,
                quantity=quantity,
                metadata={
                    'symbol': symbol,
                    'rsi_value': current_rsi,
                    'timestamp': market_data.get('timestamp', 0),
                    'market_data': market_data
                }
            )
            
            # Record signal for performance tracking
            self._record_signal(signal)
            
            logger.info(f"RSI signal for {symbol}: {action} (RSI: {current_rsi:.2f}, "
                       f"confidence: {confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating RSI signal for {symbol}: {e}")
            return None
    
    def _analyze_rsi(self, rsi_value: float) -> tuple[str, float]:
        """
        Analyze RSI value and determine action and confidence.
        
        Args:
            rsi_value: Current RSI value
            
        Returns:
            Tuple of (action, confidence)
        """
        overbought = self.parameters['rsi_overbought']
        oversold = self.parameters['rsi_oversold']
        extreme_overbought = self.parameters['rsi_extreme_overbought']
        extreme_oversold = self.parameters['rsi_extreme_oversold']
        min_confidence = self.parameters['min_confidence']
        max_confidence = self.parameters['max_confidence']
        
        # Determine action
        if rsi_value <= oversold:
            action = 'buy'
        elif rsi_value >= overbought:
            action = 'sell'
        else:
            return 'hold', 0.0
        
        # Calculate confidence based on RSI extremity
        if action == 'buy':
            if rsi_value <= extreme_oversold:
                # Extreme oversold - high confidence
                confidence = max_confidence
            else:
                # Normal oversold - moderate confidence
                confidence = min_confidence + (oversold - rsi_value) / (oversold - extreme_oversold) * (max_confidence - min_confidence)
        
        else:  # action == 'sell'
            if rsi_value >= extreme_overbought:
                # Extreme overbought - high confidence
                confidence = max_confidence
            else:
                # Normal overbought - moderate confidence
                confidence = min_confidence + (rsi_value - overbought) / (extreme_overbought - overbought) * (max_confidence - min_confidence)
        
        # Ensure confidence is within bounds
        confidence = max(min_confidence, min(max_confidence, confidence))
        
        return action, confidence
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RSI strategy performance metrics."""
        if not self.performance_history:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0.0,
                'avg_rsi': 0.0
            }
        
        # Calculate metrics
        total_signals = len(self.performance_history)
        buy_signals = sum(1 for s in self.performance_history if s['action'] == 'buy')
        sell_signals = sum(1 for s in self.performance_history if s['action'] == 'sell')
        avg_confidence = np.mean([s['confidence'] for s in self.performance_history])
        
        # Calculate average RSI for signals
        rsi_values = [s['metadata'].get('rsi_value', 50) for s in self.performance_history 
                     if 'metadata' in s and 'rsi_value' in s['metadata']]
        avg_rsi = np.mean(rsi_values) if rsi_values else 50.0
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'avg_rsi': avg_rsi,
            'strategy_name': 'rsi'
        }
    
    def get_rsi_analysis(self, market_data: Dict) -> Dict[str, Any]:
        """
        Get detailed RSI analysis for debugging and monitoring.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary with RSI analysis
        """
        if not self._validate_market_data(market_data):
            return {}
        
        try:
            ohlcv = market_data['ohlcv']
            rsi_values = self._calculate_rsi(ohlcv['close'].values, self.parameters['rsi_period'])
            
            if len(rsi_values) == 0:
                return {}
            
            current_rsi = rsi_values[-1]
            action, confidence = self._analyze_rsi(current_rsi)
            
            return {
                'current_rsi': current_rsi,
                'action': action,
                'confidence': confidence,
                'overbought_threshold': self.parameters['rsi_overbought'],
                'oversold_threshold': self.parameters['rsi_oversold'],
                'extreme_overbought': self.parameters['rsi_extreme_overbought'],
                'extreme_oversold': self.parameters['rsi_extreme_oversold'],
                'rsi_trend': self._get_rsi_trend(rsi_values)
            }
            
        except Exception as e:
            logger.error(f"Error in RSI analysis: {e}")
            return {}
    
    def _get_rsi_trend(self, rsi_values: np.ndarray) -> str:
        """Determine RSI trend based on recent values."""
        if len(rsi_values) < 5:
            return 'unknown'
        
        recent_rsi = rsi_values[-5:]
        
        # Calculate trend
        if np.all(np.diff(recent_rsi) > 0):
            return 'rising'
        elif np.all(np.diff(recent_rsi) < 0):
            return 'falling'
        else:
            return 'sideways' 