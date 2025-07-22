"""
Competition Trading Strategy

This strategy is specifically designed for hackathon competitions where
the goal is to maximize returns in a short time period.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from loguru import logger

from .base_strategy import BaseStrategy
from .types import StrategySignal, SignalType


class CompetitionStrategy(BaseStrategy):
    """
    Aggressive trading strategy for hackathon competitions.
    
    This strategy:
    - Takes more risks for higher potential returns
    - Uses shorter timeframes for faster signals
    - Trades more frequently
    - Uses momentum and volatility for quick profits
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Get default parameters
        default_params = self._get_default_parameters()
        
        # Merge with provided config
        if 'parameters' not in config:
            config['parameters'] = {}
        config['parameters'] = {**default_params, **config['parameters']}
        
        # Call parent constructor
        super().__init__(config)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default competition strategy parameters."""
        return {
            # RSI parameters - Very aggressive
            'rsi_period': 10,  # Shorter period for faster signals
            'rsi_overbought': 80,  # Higher overbought
            'rsi_oversold': 20,    # Lower oversold
            
            # MACD parameters - Faster
            'macd_fast': 8,   # Faster MACD
            'macd_slow': 21,  # Faster MACD
            'macd_signal': 5, # Faster signal
            
            # Momentum parameters
            'momentum_period': 5,
            'momentum_threshold': 0.001,  # Small price changes trigger trades
            
            # Volatility parameters
            'volatility_period': 10,
            'volatility_threshold': 0.005,  # Lower threshold for more trades
            
            # Signal parameters - Very aggressive
            'min_confidence': 0.2,      # Very low minimum confidence
            'max_confidence': 0.99,     # Very high maximum confidence
            'always_trade': True,       # Always generate a signal
            'momentum_weight': 0.4,     # Weight for momentum signals
            'volatility_weight': 0.3,   # Weight for volatility signals
            'technical_weight': 0.3     # Weight for technical indicators
        }
    
    async def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Generate an aggressive trading signal for competition.
        """
        try:
            # Extract OHLCV data
            ohlcv = market_data.get('ohlcv')
            if ohlcv is None or len(ohlcv) < 30:
                logger.debug(f"Insufficient OHLCV data for {symbol}")
                return None

            # Calculate indicators
            rsi = self._calculate_rsi(ohlcv['close'], self.parameters.get('rsi_period', 10))
            macd_line, signal_line, histogram = self._calculate_macd(
                ohlcv['close'],
                self.parameters.get('macd_fast', 8),
                self.parameters.get('macd_slow', 21),
                self.parameters.get('macd_signal', 5)
            )
            
            # Calculate momentum
            momentum = self._calculate_momentum(ohlcv['close'], self.parameters.get('momentum_period', 5))
            
            # Calculate volatility
            volatility = self._calculate_volatility(ohlcv['close'], self.parameters.get('volatility_period', 10))
            
            current_price = ohlcv['close'].iloc[-1]
            current_rsi = rsi[-1] if len(rsi) > 0 else 50
            current_macd = macd_line[-1] if len(macd_line) > 0 else 0
            current_signal = signal_line[-1] if len(signal_line) > 0 else 0
            current_momentum = momentum[-1] if len(momentum) > 0 else 0
            current_volatility = volatility[-1] if len(volatility) > 0 else 0

            # Aggressive decision logic
            action = 'hold'
            confidence = 0.0
            signal_strength = 0.0

            # Technical indicators (30% weight)
            tech_score = 0
            if current_rsi < self.parameters.get('rsi_oversold', 20):
                tech_score += 1
            elif current_rsi > self.parameters.get('rsi_overbought', 80):
                tech_score -= 1
            
            if current_macd > current_signal:
                tech_score += 1
            else:
                tech_score -= 1

            # Momentum signals (40% weight)
            momentum_score = 0
            if current_momentum > self.parameters.get('momentum_threshold', 0.001):
                momentum_score = 1
            elif current_momentum < -self.parameters.get('momentum_threshold', 0.001):
                momentum_score = -1

            # Volatility signals (30% weight)
            volatility_score = 0
            if current_volatility > self.parameters.get('volatility_threshold', 0.005):
                # High volatility - take advantage of swings
                if tech_score > 0:
                    volatility_score = 1
                elif tech_score < 0:
                    volatility_score = -1

            # Combine signals with weights
            signal_strength = (
                tech_score * self.parameters.get('technical_weight', 0.3) +
                momentum_score * self.parameters.get('momentum_weight', 0.4) +
                volatility_score * self.parameters.get('volatility_weight', 0.3)
            )

            # Generate action based on signal strength
            min_conf = self.parameters.get('min_confidence', 0.2)
            max_conf = self.parameters.get('max_confidence', 0.99)

            if signal_strength > 0.1:
                action = 'buy'
                confidence = min_conf + (max_conf - min_conf) * min(abs(signal_strength), 1.0)
            elif signal_strength < -0.1:
                action = 'sell'
                confidence = min_conf + (max_conf - min_conf) * min(abs(signal_strength), 1.0)
            else:
                # Even for weak signals, take action if always_trade is enabled
                if self.parameters.get('always_trade', True):
                    action = 'buy' if signal_strength > 0 else 'sell'
                    confidence = min_conf + (max_conf - min_conf) * 0.3
                else:
                    return None

            logger.info(f"Competition signal for {symbol}: {action} (confidence: {confidence:.2f}, strength: {signal_strength:.2f})")

            # Build signal object
            signal_type = SignalType.BUY if action == 'buy' else SignalType.SELL
            signal = StrategySignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                quantity=0.1,  # Will be adjusted by risk manager
                timestamp=market_data.get('timestamp'),
                strategy_name='competition',
                metadata={
                    'indicators': {
                        'rsi': current_rsi,
                        'macd': current_macd,
                        'signal': current_signal,
                        'momentum': current_momentum,
                        'volatility': current_volatility
                    },
                    'signal_analysis': {
                        'action': action,
                        'confidence': confidence,
                        'signal_strength': signal_strength,
                        'tech_score': tech_score,
                        'momentum_score': momentum_score,
                        'volatility_score': volatility_score
                    },
                    'market_data': market_data
                }
            )
            return signal

        except Exception as e:
            logger.error(f"Error generating competition signal for {symbol}: {e}")
            return None

    def _calculate_momentum(self, prices: pd.Series, period: int) -> np.ndarray:
        """Calculate price momentum."""
        try:
            if len(prices) < period:
                return np.array([])
            
            momentum = prices.diff(period).values
            return momentum
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return np.array([])

    def _calculate_volatility(self, prices: pd.Series, period: int) -> np.ndarray:
        """Calculate price volatility."""
        try:
            if len(prices) < period:
                return np.array([])
            
            returns = prices.pct_change().values
            volatility = pd.Series(returns).rolling(window=period).std().values
            return volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return np.array([]) 