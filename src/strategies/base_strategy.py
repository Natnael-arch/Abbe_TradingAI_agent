"""
Base Strategy Class

This module provides the base class for all trading strategies.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from loguru import logger

from .types import StrategySignal, SignalType


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This class provides:
    - Common interface for all strategies
    - Basic technical indicator calculations
    - Signal generation framework
    - Risk management integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 1.0)
        
        # Strategy parameters
        self.parameters = config.get('parameters', {})
        
        # Performance tracking
        self.signals_generated = 0
        self.successful_signals = 0
        
        logger.info(f"Initialized strategy: {self.name}")
    
    @abstractmethod
    async def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Generate a trading signal for the given symbol and market data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            market_data: Market data including OHLCV, indicators, etc.
            
        Returns:
            StrategySignal if a signal is generated, None otherwise
        """
        pass
    
    def calculate_indicators(self, market_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate technical indicators for the given market data.
        
        Args:
            market_data: Market data with OHLCV information
            
        Returns:
            Dictionary of calculated indicators
        """
        try:
            # Extract OHLCV data
            ohlcv = market_data.get('ohlcv', [])
            if not ohlcv:
                return {}
            
            # Convert to numpy arrays
            data = np.array(ohlcv)
            if len(data) < 50:  # Need minimum data for indicators
                return {}
            
            # Extract price data
            close_prices = data[:, 4].astype(float)  # Close prices
            high_prices = data[:, 2].astype(float)   # High prices
            low_prices = data[:, 3].astype(float)    # Low prices
            volumes = data[:, 5].astype(float)       # Volumes
            
            indicators = {}
            
            # Calculate RSI
            indicators['rsi'] = self._calculate_rsi(close_prices)
            
            # Calculate MACD
            macd_data = self._calculate_macd(close_prices)
            indicators.update(macd_data)
            
            # Calculate moving averages
            indicators['sma_20'] = self._calculate_sma(close_prices, 20)
            indicators['sma_50'] = self._calculate_sma(close_prices, 50)
            indicators['ema_12'] = self._calculate_ema(close_prices, 12)
            indicators['ema_26'] = self._calculate_ema(close_prices, 26)
            
            # Calculate Bollinger Bands
            bb_data = self._calculate_bollinger_bands(close_prices, 20, 2)
            indicators.update(bb_data)
            
            # Calculate volume indicators
            indicators['volume_sma'] = self._calculate_sma(volumes, 20)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (RSI)."""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = pd.Series(gains).rolling(window=period).mean().values
            avg_losses = pd.Series(losses).rolling(window=period).mean().values
            
            rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return np.zeros(len(prices))
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            ema_fast = pd.Series(prices).ewm(span=fast).mean().values
            ema_slow = pd.Series(prices).ewm(span=slow).mean().values
            
            macd_line = ema_fast - ema_slow
            signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
            histogram = macd_line - signal_line
            
            return {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                'macd_line': np.zeros(len(prices)),
                'signal_line': np.zeros(len(prices)),
                'histogram': np.zeros(len(prices))
            }
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average (SMA)."""
        try:
            return pd.Series(prices).rolling(window=period).mean().values
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return np.zeros(len(prices))
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average (EMA)."""
        try:
            return pd.Series(prices).ewm(span=period).mean().values
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return np.zeros(len(prices))
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, 
                                  std_dev: float = 2) -> Dict[str, np.ndarray]:
        """Calculate Bollinger Bands."""
        try:
            sma = pd.Series(prices).rolling(window=period).mean().values
            std = pd.Series(prices).rolling(window=period).std().values
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'bb_upper': upper_band,
                'bb_middle': sma,
                'bb_lower': lower_band
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'bb_upper': np.zeros(len(prices)),
                'bb_middle': np.zeros(len(prices)),
                'bb_lower': np.zeros(len(prices))
            }
    
    def validate_signal(self, signal: StrategySignal) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: The signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        if not signal:
            return False
        
        # Check basic requirements
        if signal.confidence < 0 or signal.confidence > 1:
            logger.warning(f"Invalid confidence level: {signal.confidence}")
            return False
        
        if signal.price <= 0:
            logger.warning(f"Invalid price: {signal.price}")
            return False
        
        if signal.quantity <= 0:
            logger.warning(f"Invalid quantity: {signal.quantity}")
            return False
        
        return True
    
    def update_performance(self, signal: StrategySignal, success: bool):
        """
        Update strategy performance metrics.
        
        Args:
            signal: The signal that was executed
            success: Whether the signal was successful
        """
        self.signals_generated += 1
        if success:
            self.successful_signals += 1
        
        success_rate = self.successful_signals / self.signals_generated if self.signals_generated > 0 else 0
        logger.info(f"Strategy {self.name} performance: {success_rate:.2%} success rate")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary."""
        success_rate = self.successful_signals / self.signals_generated if self.signals_generated > 0 else 0
        
        return {
            'name': self.name,
            'enabled': self.enabled,
            'weight': self.weight,
            'signals_generated': self.signals_generated,
            'successful_signals': self.successful_signals,
            'success_rate': success_rate
        } 