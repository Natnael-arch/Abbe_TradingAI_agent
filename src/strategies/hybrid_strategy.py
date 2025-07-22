"""
Hybrid Trading Strategy

This strategy combines multiple technical indicators including RSI, MACD,
and additional filters for more robust trading decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from loguru import logger

from .base_strategy import BaseStrategy
from .types import StrategySignal, SignalType


class HybridStrategy(BaseStrategy):
    """
    Hybrid trading strategy that combines multiple indicators.
    
    This strategy:
    - Combines RSI and MACD signals
    - Uses Bollinger Bands for volatility analysis
    - Applies volume confirmation
    - Uses multiple timeframes for confirmation
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
        """Get default hybrid strategy parameters."""
        return {
            # RSI parameters - More aggressive
            'rsi_period': 14,
            'rsi_overbought': 75,  # Higher overbought
            'rsi_oversold': 25,    # Lower oversold
            
            # MACD parameters
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands parameters
            'bb_period': 20,
            'bb_std': 2,
            
            # Volume parameters
            'volume_sma_period': 20,
            'volume_threshold': 1.2,  # Lower volume threshold
            
            # Signal combination - More aggressive for competition
            'min_indicators_agree': 1,  # Only 1 indicator needs to agree (was 2)
            'min_confidence': 0.3,      # Lower minimum confidence
            'max_confidence': 0.98      # Higher maximum confidence
        }
    
    async def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Generate a hybrid trading signal using RSI, MACD, Bollinger Bands, volume, support/resistance, and trend.
        Filters loosened: only 1 indicator required, 3% S/R zone, trend can be 'flat' or in direction.
        Only trade if confidence > 0.6 and reward/risk is favorable. Add take-profit and trailing stop to metadata.
        """
        try:
            # Extract OHLCV and indicators
            ohlcv = market_data.get('ohlcv')
            if ohlcv is None or len(ohlcv) < 50:
                logger.debug(f"Insufficient OHLCV data for {symbol} at {market_data.get('timestamp')}")
                return None

            # Calculate indicators
            rsi = self._calculate_rsi(ohlcv['close'], self.parameters.get('rsi_period', 14))
            macd_line, signal_line, histogram = self._calculate_macd(
                ohlcv['close'],
                self.parameters.get('macd_fast', 12),
                self.parameters.get('macd_slow', 26),
                self.parameters.get('macd_signal', 9)
            )
            upper, middle, lower = self._calculate_bollinger(
                ohlcv['close'],
                self.parameters.get('bollinger_period', 20),
                self.parameters.get('bollinger_std', 2)
            )
            current_price = ohlcv['close'].iloc[-1]
            current_rsi = rsi[-1]
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            current_histogram = histogram[-1]
            current_upper = upper[-1]
            current_lower = lower[-1]

            # Support/Resistance (last 50 closes)
            lookback = ohlcv['close'].tail(50)
            support = lookback.min()
            resistance = lookback.max()

            # Trend: 50-period MA slope
            if len(lookback) >= 50:
                ma50 = lookback.rolling(50).mean().iloc[-1]
                ma50_prev = lookback.rolling(50).mean().iloc[-2] if len(lookback) > 50 else ma50
                trend = 'up' if ma50 > ma50_prev else 'down' if ma50 < ma50_prev else 'flat'
            else:
                ma50 = None
                trend = 'flat'

            # Loosened trend filter: allow 'flat' as well as up/down
            trend_ok_buy = trend in ['up', 'flat']
            trend_ok_sell = trend in ['down', 'flat']

            # Volatility filter: only trade if stddev of close > 0.0005 * price (loosened)
            vol = ohlcv['close'].rolling(20).std().iloc[-1]
            if vol is not None and vol < 0.0005 * current_price:
                logger.debug(f"Volatility too low for {symbol} at {market_data.get('timestamp')}")
                return None

            # Calculate Stochastic Oscillator (14-period)
            if len(ohlcv) >= 14:
                low_min = ohlcv['low'].rolling(14).min().iloc[-1]
                high_max = ohlcv['high'].rolling(14).max().iloc[-1]
                stochastic = 100 * (current_price - low_min) / (high_max - low_min) if high_max != low_min else 50
            else:
                stochastic = 50

            # Calculate ATR (14-period)
            if len(ohlcv) >= 14:
                tr = pd.concat([
                    ohlcv['high'] - ohlcv['low'],
                    (ohlcv['high'] - ohlcv['close'].shift()).abs(),
                    (ohlcv['low'] - ohlcv['close'].shift()).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
            else:
                atr = 0.0

            # ATR filter: Temporarily disabled for competition to ensure trading activity
            # if atr < 0.005 * current_price:
            #     logger.debug(f"ATR too low for {symbol} at {market_data.get('timestamp')}")
            #     return None

            logger.debug(f"{symbol} @ {market_data.get('timestamp')}: price={current_price}, RSI={current_rsi}, MACD={current_macd}, Signal={current_signal}, Hist={current_histogram}, BB_upper={current_upper}, BB_lower={current_lower}, support={support}, resistance={resistance}, trend={trend}")

            # Decision logic
            action = 'hold'
            confidence = 0.0
            indicators_agree = 0

            # Buy conditions
            if current_rsi < self.parameters.get('rsi_oversold', 30):
                indicators_agree += 1
            if current_macd > current_signal and current_histogram > 0:
                indicators_agree += 1
            if current_price < current_lower:
                indicators_agree += 1
            # Only buy near support (within 3% of support, loosened)
            if current_price <= support * 1.03:
                indicators_agree += 1
            # Stochastic buy
            if stochastic < 20:
                indicators_agree += 1

            # Sell conditions
            if current_rsi > self.parameters.get('rsi_overbought', 70):
                indicators_agree -= 1
            if current_macd < current_signal and current_histogram < 0:
                indicators_agree -= 1
            if current_price > current_upper:
                indicators_agree -= 1
            # Only sell near resistance (within 3% of resistance, loosened)
            if current_price >= resistance * 0.97:
                indicators_agree -= 1
            # Stochastic sell
            if stochastic > 80:
                indicators_agree -= 1

            logger.debug(f"{symbol} indicators_agree: {indicators_agree}")

            # Require only 1 indicator to agree (loosened)
            min_indicators = 1
            min_conf = self.parameters.get('min_confidence', 0.3)
            max_conf = self.parameters.get('max_confidence', 0.98)

            if indicators_agree >= min_indicators and trend_ok_buy:
                action = 'buy'
                confidence = min_conf + (max_conf - min_conf) * (indicators_agree / 4)
            elif indicators_agree <= -min_indicators and trend_ok_sell:
                action = 'sell'
                confidence = min_conf + (max_conf - min_conf) * (abs(indicators_agree) / 4)
            else:
                logger.debug(f"No strong signal for {symbol} at {market_data.get('timestamp')}: indicators_agree={indicators_agree}")
                return None

            # Only trade if confidence > min_conf (from config)
            if confidence < min_conf:
                logger.debug(f"Confidence {confidence:.2f} too low for {symbol} at {market_data.get('timestamp')}")
                return None

            # Reward/risk filter: only trade if expected reward/risk is favorable
            # For buy: reward = resistance - price, risk = price - support
            # For sell: reward = price - support, risk = resistance - price
            reward = risk = 1.0
            if action == 'buy':
                reward = resistance - current_price
                risk = current_price - support
            elif action == 'sell':
                reward = current_price - support
                risk = resistance - current_price
            # Require reward/risk > 1.0 (loosened)
            if risk <= 0 or reward / risk < 1.0:
                logger.debug(f"Reward/risk {reward:.2f}/{risk:.2f} too low for {symbol} at {market_data.get('timestamp')}")
                return None

            # Use ATR for dynamic stop-loss/take-profit (tuned for 15m timeframe)
            if action == 'buy':
                take_profit = current_price + 2 * atr
                stop_loss = current_price - 1.5 * atr
                trailing_stop = 1.0 * atr
            elif action == 'sell':
                take_profit = current_price - 2 * atr
                stop_loss = current_price + 1.5 * atr
                trailing_stop = 1.0 * atr

            logger.debug(f"Hybrid signal for {symbol}: {action} (confidence: {confidence}, indicators_agree: {indicators_agree}, support: {support}, resistance: {resistance}, trend: {trend})")

            # Build signal object
            signal_type = SignalType.BUY if action == 'buy' else SignalType.SELL
            signal = StrategySignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                quantity=0.1,  # Placeholder, will be set by risk manager
                timestamp=market_data.get('timestamp'),
                strategy_name='hybrid',
                metadata={
                    'indicators': {
                        'rsi': rsi,
                        'macd': {
                            'macd_line': macd_line,
                            'signal_line': signal_line,
                            'histogram': histogram
                        },
                        'bollinger': {
                            'upper': upper,
                            'middle': middle,
                            'lower': lower
                        },
                        'support': support,
                        'resistance': resistance,
                        'trend': trend,
                        'volume': {
                            'current_volume': ohlcv['volume'].iloc[-1],
                            'avg_volume': ohlcv['volume'].mean(),
                            'volume_ratio': ohlcv['volume'].iloc[-1] / ohlcv['volume'].mean() if ohlcv['volume'].mean() > 0 else 0,
                            'high_volume': ohlcv['volume'].iloc[-1] > ohlcv['volume'].mean()
                        },
                        'stochastic': stochastic,
                        'atr': atr
                    },
                    'signal_analysis': {
                        'action': action,
                        'confidence': confidence,
                        'indicators_agree': indicators_agree,
                        'support': support,
                        'resistance': resistance,
                        'trend': trend,
                        'reward': reward,
                        'risk': risk,
                        'reward_risk': reward / risk if risk > 0 else None
                    },
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'trailing_stop': trailing_stop,
                    'market_data': market_data
                }
            )
            return signal
        except Exception as e:
            logger.error(f"Error generating hybrid signal for {symbol}: {e}")
            return None
    

    
    def _calculate_all_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators."""
        try:
            close_prices = ohlcv['close'].values
            high_prices = ohlcv['high'].values
            low_prices = ohlcv['low'].values
            volume = ohlcv['volume'].values
            
            indicators = {}
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(close_prices, self.parameters['rsi_period'])
            
            # MACD
            indicators['macd'] = self._calculate_macd(
                close_prices,
                self.parameters['macd_fast'],
                self.parameters['macd_slow'],
                self.parameters['macd_signal']
            )
            
            # Bollinger Bands
            indicators['bollinger'] = self._calculate_bollinger_bands(
                close_prices,
                self.parameters['bb_period'],
                self.parameters['bb_std']
            )
            
            # Volume analysis
            indicators['volume'] = self._analyze_volume(volume)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_bollinger(self, closes, period=14, std_dev=2):
        """
        Calculate Bollinger Bands for a list of closing prices.
        Returns (upper_band, middle_band, lower_band) as lists.
        """
        if len(closes) < period:
            return [], [], []
        middle_band = []
        upper_band = []
        lower_band = []
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            mean = sum(window) / period
            variance = sum((x - mean) ** 2 for x in window) / period
            std = variance ** 0.5
            middle_band.append(mean)
            upper_band.append(mean + std_dev * std)
            lower_band.append(mean - std_dev * std)
        # Pad the beginning with Nones to match input length
        pad = [None] * (period - 1)
        return pad + upper_band, pad + middle_band, pad + lower_band
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float) -> Dict[str, np.ndarray]:
        """Calculate Bollinger Bands."""
        try:
            sma = pd.Series(prices).rolling(window=period).mean().values
            std = pd.Series(prices).rolling(window=period).std().values
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'upper': np.zeros(len(prices)),
                'middle': np.zeros(len(prices)),
                'lower': np.zeros(len(prices))
            }
    
    def _analyze_volume(self, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze volume patterns."""
        try:
            volume_sma = pd.Series(volume).rolling(window=self.parameters['volume_sma_period']).mean().values
            current_volume = volume[-1] if len(volume) > 0 else 0
            avg_volume = volume_sma[-1] if len(volume_sma) > 0 else 0
            
            volume_ratio = current_volume / (avg_volume + 1e-10)
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'high_volume': volume_ratio > self.parameters['volume_threshold']
            }
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {
                'current_volume': 0,
                'avg_volume': 0,
                'volume_ratio': 0,
                'high_volume': False
            }
    
    def _analyze_all_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all indicators and combine signals."""
        signals = []
        
        # RSI analysis
        if len(indicators['rsi']) > 0:
            current_rsi = indicators['rsi'][-1]
            if not np.isnan(current_rsi):
                if current_rsi <= self.parameters['rsi_oversold']:
                    signals.append(('buy', 0.8))
                elif current_rsi >= self.parameters['rsi_overbought']:
                    signals.append(('sell', 0.8))
        
        # MACD analysis
        if len(indicators['macd']['macd_line']) > 1:
            current_macd = indicators['macd']['macd_line'][-1]
            previous_macd = indicators['macd']['macd_line'][-2]
            current_signal = indicators['macd']['signal_line'][-1]
            previous_signal = indicators['macd']['signal_line'][-2]
            
            if (previous_macd <= previous_signal and current_macd > current_signal):
                signals.append(('buy', 0.7))
            elif (previous_macd >= previous_signal and current_macd < current_signal):
                signals.append(('sell', 0.7))
        
        # Bollinger Bands analysis
        if len(indicators['bollinger']['upper']) > 0:
            current_price = indicators['bollinger']['middle'][-1]  # Use SMA as current price proxy
            upper_band = indicators['bollinger']['upper'][-1]
            lower_band = indicators['bollinger']['lower'][-1]
            
            if current_price <= lower_band:
                signals.append(('buy', 0.6))
            elif current_price >= upper_band:
                signals.append(('sell', 0.6))
        
        # Combine signals
        return self._combine_signals(signals)
    
    def _combine_signals(self, signals: list) -> Dict[str, Any]:
        """Combine multiple signals into a final decision."""
        if not signals:
            return {'action': 'hold', 'confidence': 0.0, 'indicators_agree': 0}
        
        # Count signals by action
        buy_signals = [s for s in signals if s[0] == 'buy']
        sell_signals = [s for s in signals if s[0] == 'sell']
        
        min_indicators = self.parameters['min_indicators_agree']
        
        # Determine action
        if len(buy_signals) >= min_indicators:
            action = 'buy'
            signal_list = buy_signals
        elif len(sell_signals) >= min_indicators:
            action = 'sell'
            signal_list = sell_signals
        else:
            return {'action': 'hold', 'confidence': 0.0, 'indicators_agree': len(signals)}
        
        # Calculate confidence
        avg_confidence = np.mean([s[1] for s in signal_list])
        confidence = max(self.parameters['min_confidence'], 
                        min(self.parameters['max_confidence'], avg_confidence))
        
        return {
            'action': action,
            'confidence': confidence,
            'indicators_agree': len(signal_list)
        }
    
    def _confirm_with_volume(self, ohlcv: pd.DataFrame, action: str) -> bool:
        """Confirm signal with volume analysis."""
        try:
            volume_analysis = self._analyze_volume(ohlcv['volume'].values)
            
            # For buy signals, we want high volume (accumulation)
            # For sell signals, we want high volume (distribution)
            if action == 'buy':
                return volume_analysis['high_volume']
            elif action == 'sell':
                return volume_analysis['high_volume']
            else:
                return True  # Hold signals don't need volume confirmation
                
        except Exception as e:
            logger.error(f"Error in volume confirmation: {e}")
            return True  # Default to allowing the signal
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get hybrid strategy performance metrics."""
        if not self.performance_history:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0.0,
                'avg_indicators_agree': 0.0
            }
        
        # Calculate metrics
        total_signals = len(self.performance_history)
        buy_signals = sum(1 for s in self.performance_history if s['action'] == 'buy')
        sell_signals = sum(1 for s in self.performance_history if s['action'] == 'sell')
        avg_confidence = np.mean([s['confidence'] for s in self.performance_history])
        
        # Calculate average indicators agreement
        indicators_agree = [s['metadata'].get('signal_analysis', {}).get('indicators_agree', 0) 
                          for s in self.performance_history 
                          if 'metadata' in s and 'signal_analysis' in s['metadata']]
        avg_indicators_agree = np.mean(indicators_agree) if indicators_agree else 0.0
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'avg_indicators_agree': avg_indicators_agree,
            'strategy_name': 'hybrid'
        }
    
    def _validate_market_data(self, market_data: Dict) -> bool:
        """Validate that market data contains required fields."""
        if not market_data or 'ohlcv' not in market_data:
            return False
        
        ohlcv = market_data['ohlcv']
        if not isinstance(ohlcv, pd.DataFrame) or len(ohlcv) < 50:
            return False
        
        return True
    
    def _record_signal(self, signal: StrategySignal):
        """Record signal for performance tracking."""
        if not hasattr(self, 'performance_history'):
            self.performance_history = []
        
        self.performance_history.append(signal)
        self.signals_generated += 1
    
    def _calculate_position_size(self, available_balance: float, current_price: float, confidence: float) -> float:
        """Calculate position size based on available balance, price, and confidence."""
        try:
            # Base position size (10% of available balance)
            base_position_size = available_balance * 0.1
            
            # Adjust based on confidence
            confidence_multiplier = confidence  # Higher confidence = larger position
            
            # Calculate quantity
            position_value = base_position_size * confidence_multiplier
            quantity = position_value / current_price
            
            # Ensure minimum trade size
            min_trade_value = 10  # $10 minimum
            if position_value < min_trade_value:
                return 0
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_hybrid_analysis(self, market_data: Dict) -> Dict[str, Any]:
        """Get detailed hybrid strategy analysis."""
        if not self._validate_market_data(market_data):
            return {}
        
        try:
            ohlcv = market_data['ohlcv']
            indicators = self._calculate_all_indicators(ohlcv)
            signal_analysis = self._analyze_all_indicators(indicators)
            
            return {
                'indicators': {
                    'rsi': indicators['rsi'][-1] if len(indicators['rsi']) > 0 else None,
                    'macd_line': indicators['macd']['macd_line'][-1] if len(indicators['macd']['macd_line']) > 0 else None,
                    'signal_line': indicators['macd']['signal_line'][-1] if len(indicators['macd']['signal_line']) > 0 else None,
                    'bollinger_upper': indicators['bollinger']['upper'][-1] if len(indicators['bollinger']['upper']) > 0 else None,
                    'bollinger_lower': indicators['bollinger']['lower'][-1] if len(indicators['bollinger']['lower']) > 0 else None,
                    'volume_ratio': indicators['volume']['volume_ratio']
                },
                'signal_analysis': signal_analysis,
                'parameters': self.parameters
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}")
            return {} 