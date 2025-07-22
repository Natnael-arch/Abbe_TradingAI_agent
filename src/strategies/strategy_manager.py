"""
Strategy Manager

This module manages multiple trading strategies and combines their signals.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger

from .base_strategy import BaseStrategy
from .types import StrategySignal, SignalType
from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy
from .hybrid_strategy import HybridStrategy
from .gaia_llm_strategy import GaiaLLMStrategy
from .competition_strategy import CompetitionStrategy
from ..utils.config import Config


class StrategyManager:
    """
    Manages multiple trading strategies and combines their signals.
    
    This class:
    - Loads and initializes multiple strategies
    - Generates signals from all active strategies
    - Combines signals using weighted voting
    - Integrates with Gaia for AI-enhanced decisions
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.strategies = {}
        self.gaia_client = None
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Initialize Gaia client if configured
        self._initialize_gaia()
        
        logger.info(f"Strategy manager initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self):
        """Initialize all configured strategies."""
        strategy_config = self.config.get_strategy_config()
        params = strategy_config.parameters or {}

        # If using ensemble/multi-strategy
        if strategy_config.name == 'ensemble' and 'strategies' in params:
            for strat in params['strategies']:
                strat_name = strat['name']
                # Prevent instantiation of BaseStrategy
                if strat_name.lower() in ['base', 'basestrategy']:
                    logger.warning("Skipping abstract BaseStrategy in strategy list.")
                    continue
                if strat_name == 'gaia_llm':
                    self.strategies['gaia_llm'] = GaiaLLMStrategy(params)
                elif strat_name == 'hybrid':
                    self.strategies['hybrid'] = HybridStrategy(params)
                elif strat_name == 'rsi':
                    self.strategies['rsi'] = RSIStrategy(params)
                elif strat_name == 'macd':
                    self.strategies['macd'] = MACDStrategy(params)
                # Add more as needed
        else:
            # Fallback to single strategy logic (as before)
            if strategy_config.name.lower() in ['base', 'basestrategy']:
                logger.error("Cannot instantiate abstract BaseStrategy. Please check your config.")
                raise ValueError("Cannot instantiate abstract BaseStrategy.")
            if strategy_config.name == 'gaia_llm':
                self.strategies['gaia_llm'] = GaiaLLMStrategy(params)
            elif strategy_config.name == 'rsi':
                self.strategies['rsi'] = RSIStrategy(params)
            elif strategy_config.name == 'macd':
                self.strategies['macd'] = MACDStrategy(params)
            elif strategy_config.name == 'hybrid':
                self.strategies['hybrid'] = HybridStrategy(params)
            elif strategy_config.name == 'competition':
                self.strategies['competition'] = CompetitionStrategy(params)
            else:
                logger.warning(f"Unknown strategy '{strategy_config.name}', using competition strategy")
                self.strategies['competition'] = CompetitionStrategy(params)
    
    def _initialize_gaia(self):
        """Initialize Gaia client for AI inference."""
        try:
            gaia_config = self.config.get_gaia_config()
            if gaia_config.enabled and gaia_config.api_key:
                from ..ai.gaia_client import GaiaClient
                self.gaia_client = GaiaClient(gaia_config)
                logger.info("Gaia client initialized successfully")
                
                # Test connection
                asyncio.create_task(self._test_gaia_connection())
            else:
                logger.info("Gaia integration disabled or not configured")
        except Exception as e:
            logger.warning(f"Failed to initialize Gaia client: {e}")
    
    async def _test_gaia_connection(self):
        """Test Gaia connection asynchronously."""
        try:
            gaia_config = self.config.get_gaia_config()
            if self.gaia_client and gaia_config.enabled:
                # Use async with to correctly initialize the session
                from ..ai.gaia_client import GaiaClient
                async with GaiaClient(gaia_config) as client:
                    is_connected = await client.test_connection()
                if is_connected:
                    logger.info("Gaia AI connection test successful")
                else:
                    logger.warning("Gaia AI connection test failed")
        except Exception as e:
            logger.error(f"Error testing Gaia connection: {e}")
    
    async def generate_signals(self, symbol: str, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Generate signals from all active strategies for a symbol.
        """
        signals = []
        for name, strategy in self.strategies.items():
            signal = await strategy.generate_signal(symbol, market_data)
            if signal:
                logger.debug(f"Generated {signal.action} signal from {name} with confidence {signal.confidence}")
                signals.append(signal)
            else:
                logger.debug(f"No signal generated from {name} for {symbol} at {market_data.get('timestamp')}")
        return signals
    
    async def combine_signals(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """
        Combine multiple signals into a final decision.
        Prefer Gaia LLM if its confidence > 0.8, otherwise use existing logic.
        """
        if not signals:
            return None
        
        # Prefer Gaia LLM if confidence > 0.8
        for s in signals:
            if s.strategy_name == 'gaia_llm' and s.confidence > 0.8:
                return s

        # If only one signal, return it
        if len(signals) == 1:
            return signals[0]
        
        # Group signals by action
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        # Determine final action based on weighted voting
        buy_weight = sum(s.confidence for s in buy_signals)
        sell_weight = sum(s.confidence for s in sell_signals)
        
        # Get the most confident signal for the winning action
        if buy_weight > sell_weight and buy_signals:
            best_signal = max(buy_signals, key=lambda s: s.confidence)
            return self._create_combined_signal(best_signal, "combined_buy", buy_weight)
        elif sell_weight > buy_weight and sell_signals:
            best_signal = max(sell_signals, key=lambda s: s.confidence)
            return self._create_combined_signal(best_signal, "combined_sell", sell_weight)
        
        return None
    
    def _create_combined_signal(self, base_signal: StrategySignal, 
                               strategy_name: str, weight: float) -> StrategySignal:
        """Create a combined signal from multiple strategies."""
        # Adjust confidence based on combined weight
        adjusted_confidence = min(base_signal.confidence * (weight / len(self.strategies)), 1.0)
        
        return StrategySignal(
            symbol=base_signal.symbol,
            signal_type=base_signal.signal_type,
            confidence=adjusted_confidence,
            price=base_signal.price,
            quantity=base_signal.quantity,
            timestamp=base_signal.timestamp,
            strategy_name=strategy_name,
            metadata={
                **base_signal.metadata,
                'combined_weight': weight,
                'original_strategy': base_signal.strategy_name
            }
        )
    
    async def enhance_with_ai(self, signal: StrategySignal, market_data: Dict[str, Any]) -> StrategySignal:
        """
        Enhance signal with AI inference from Gaia.
        
        Args:
            signal: Original trading signal
            market_data: Current market data
            
        Returns:
            Enhanced signal with AI insights
        """
        if not self.gaia_client or not self.gaia_client.is_available():
            return signal
        
        try:
            # Prepare data for AI inference
            ai_data = {
                'signal': signal.to_dict(),
                'market_data': {
                    'symbol': signal.symbol,
                    'price': market_data.get('current_price', 0),
                    'volume': market_data.get('volume_24h', 0),
                    'indicators': market_data.get('indicators', {}),
                    'timestamp': market_data.get('timestamp', 0),
                    'ohlcv': market_data.get('ohlcv', {}),
                    'ticker': market_data.get('ticker', {}),
                    'orderbook': market_data.get('orderbook', {})
                }
            }
            
            # Get AI enhancement
            ai_response = await self.gaia_client.enhance_trading_signal(
                signal.to_dict(), 
                market_data
            )
            
            if ai_response and ai_response.get('enhanced_signal'):
                enhanced = ai_response['enhanced_signal']
                
                # Update signal with AI insights
                signal.confidence = enhanced.get('confidence', signal.confidence)
                signal.quantity = enhanced.get('quantity', signal.quantity)
                signal.metadata['ai_enhanced'] = True
                signal.metadata['ai_confidence'] = enhanced.get('ai_confidence', 0)
                signal.metadata['ai_reasoning'] = enhanced.get('reasoning', '')
                signal.metadata['risk_level'] = enhanced.get('risk_level', 'medium')
                
                # Add market analysis
                if 'market_analysis' in ai_response:
                    signal.metadata['market_sentiment'] = ai_response['market_analysis'].get('sentiment', 'neutral')
                    signal.metadata['market_trend'] = ai_response['market_analysis'].get('trend', 'sideways')
                    signal.metadata['volatility'] = ai_response['market_analysis'].get('volatility', 'medium')
                
                # Add risk assessment
                if 'risk_assessment' in ai_response:
                    signal.metadata['risk_assessment'] = ai_response['risk_assessment']
                
                logger.info(f"Enhanced signal with AI: confidence {signal.confidence:.2f}, "
                           f"AI confidence {enhanced.get('ai_confidence', 0):.2f}, "
                           f"reasoning: {enhanced.get('reasoning', '')[:100]}...")
            
        except Exception as e:
            logger.error(f"Error enhancing signal with AI: {e}")
        
        return signal
    
    async def get_ai_market_insights(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get AI-powered market insights for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            AI market insights
        """
        if not self.gaia_client or not self.gaia_client.is_available():
            return None
        
        try:
            insights = await self.gaia_client.get_market_insights(symbol, market_data)
            return insights
        except Exception as e:
            logger.error(f"Error getting AI market insights: {e}")
            return None
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        performance = {}
        
        for name, strategy in self.strategies.items():
            performance[name] = strategy.get_performance_summary()
        
        return performance
    
    def update_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]):
        """Update parameters for a specific strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].parameters.update(parameters)
            logger.info(f"Updated parameters for {strategy_name}: {parameters}")
        else:
            logger.warning(f"Strategy {strategy_name} not found")
    
    def enable_strategy(self, strategy_name: str, enabled: bool = True):
        """Enable or disable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Strategy {strategy_name} {status}")
        else:
            logger.warning(f"Strategy {strategy_name} not found")
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names."""
        return [name for name, strategy in self.strategies.items() if strategy.enabled] 