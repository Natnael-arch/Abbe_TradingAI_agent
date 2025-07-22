import asyncio
from typing import Dict, Any, Optional
from loguru import logger
from src.strategies.base_strategy import BaseStrategy
from src.strategies.types import StrategySignal, SignalType
from src.ai.gaia_client import GaiaClient
from src.utils.config import Config
import pandas as pd
from src.strategies.hybrid_strategy import HybridStrategy

class GaiaLLMStrategy(BaseStrategy):
    """
    Strategy that uses Gaia LLM to generate trading signals.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gaia_config = Config(config.get('config_path', 'config/config.yaml')).get_gaia_config()
        self.model = config.get('model', 'Qwen3-235B-A22B-Q4_K_M')
        self.prompt_template = config.get('prompt_template', self.default_prompt_template())

    def default_prompt_template(self) -> str:
        return (
            "You are an expert crypto trading AI. Given the following market data, recommend whether to buy, sell, or hold.\n"
            "Respond in the format: Action: <buy/sell/hold>\nConfidence: <0-1>\nReason: <short explanation>.\n"
            "Market Data:\n{market_data}\n"
        )

    def format_market_data(self, symbol: str, market_data: Dict[str, Any]) -> str:
        ohlcv = market_data.get('ohlcv')
        if ohlcv is not None:
            price = ohlcv['close'].iloc[-1]
            rsi = market_data.get('rsi', None)
            macd = market_data.get('macd', None)
            volume = ohlcv['volume'].iloc[-1]
            bb_upper = market_data.get('bb_upper', None)
            bb_lower = market_data.get('bb_lower', None)
            # Support/Resistance/Trend
            lookback = ohlcv['close'].tail(50)
            support = lookback.min() if len(lookback) > 0 else None
            resistance = lookback.max() if len(lookback) > 0 else None
            if len(lookback) >= 50:
                ma50 = lookback.rolling(50).mean().iloc[-1]
                ma50_prev = lookback.rolling(50).mean().iloc[-2] if len(lookback) > 50 else ma50
                trend = 'up' if ma50 > ma50_prev else 'down' if ma50 < ma50_prev else 'flat'
            else:
                trend = 'flat'
        else:
            price = rsi = macd = volume = bb_upper = bb_lower = support = resistance = trend = None
        return (
            f"Symbol: {symbol}\n"
            f"Price: {price}\n"
            f"RSI: {rsi}\n"
            f"MACD: {macd}\n"
            f"Volume: {volume}\n"
            f"Bollinger Bands: upper={bb_upper}, lower={bb_lower}\n"
            f"Support: {support}\n"
            f"Resistance: {resistance}\n"
            f"Trend: {trend}"
        )

    async def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        # Calculate indicators and add to market_data for Gaia
        hybrid_strategy = HybridStrategy({'name': 'hybrid'})
        ohlcv = market_data.get('ohlcv')
        if isinstance(ohlcv, pd.DataFrame) and len(ohlcv) >= 26:
            close_prices = ohlcv['close'].values
            # RSI
            if len(close_prices) >= 14:
                try:
                    rsi_val = hybrid_strategy._calculate_rsi(close_prices, 14)[-1]
                    logger.info(f"[GaiaLLM] RSI calculation result: {rsi_val} (type: {type(rsi_val)})")
                    market_data['rsi'] = float(rsi_val) if rsi_val is not None and str(rsi_val).replace('.', '').replace('-', '').isdigit() else None
                except Exception as e:
                    logger.error(f"[GaiaLLM] Error calculating RSI: {e}")
                    market_data['rsi'] = None
            else:
                logger.info(f"[GaiaLLM] Not enough data for RSI: {len(close_prices)} rows")
            # MACD
            if len(close_prices) >= 26:
                try:
                    macd_data = hybrid_strategy._calculate_macd(close_prices, 12, 26, 9)
                    macd_val = macd_data['macd_line'][-1] if len(macd_data['macd_line']) > 0 else None
                    logger.info(f"[GaiaLLM] MACD calculation result: {macd_val} (type: {type(macd_val)})")
                    market_data['macd'] = float(macd_val) if macd_val is not None and str(macd_val).replace('.', '').replace('-', '').isdigit() else None
                except Exception as e:
                    logger.error(f"[GaiaLLM] Error calculating MACD: {e}")
                    market_data['macd'] = None
            else:
                logger.info(f"[GaiaLLM] Not enough data for MACD: {len(close_prices)} rows")
            # Bollinger Bands
            if len(close_prices) >= 20:
                try:
                    upper, middle, lower = hybrid_strategy._calculate_bollinger(closes=close_prices, period=20, std_dev=2)
                    upper_val = upper[-1] if len(upper) > 0 and upper[-1] is not None else None
                    lower_val = lower[-1] if len(lower) > 0 and lower[-1] is not None else None
                    logger.info(f"[GaiaLLM] Bollinger Bands calculation result: upper={upper_val} (type: {type(upper_val)}), lower={lower_val} (type: {type(lower_val)})")
                    market_data['bb_upper'] = float(upper_val) if upper_val is not None and str(upper_val).replace('.', '').replace('-', '').isdigit() else None
                    market_data['bb_lower'] = float(lower_val) if lower_val is not None and str(lower_val).replace('.', '').replace('-', '').isdigit() else None
                except Exception as e:
                    logger.error(f"[GaiaLLM] Error calculating Bollinger Bands: {e}")
                    market_data['bb_upper'] = None
                    market_data['bb_lower'] = None
            else:
                logger.info(f"[GaiaLLM] Not enough data for Bollinger Bands: {len(close_prices)} rows")
        else:
            logger.info(f"[GaiaLLM] ohlcv is not a DataFrame or too short: type={type(ohlcv)}, len={len(ohlcv) if ohlcv is not None else 'None'}")
        
        prompt = self.prompt_template.format(market_data=self.format_market_data(symbol, market_data))
        logger.info(f"[GaiaLLMStrategy] Prompt to Gaia:\n{prompt}")
        async with GaiaClient(self.gaia_config) as client:
            params = {"model": self.model}
            response = await client.infer(prompt, params)
        if not response or 'choices' not in response or not response['choices']:
            logger.error("[GaiaLLMStrategy] No response from Gaia LLM.")
            return None
        content = response['choices'][0]['message']['content']
        logger.info(f"[GaiaLLMStrategy] Gaia LLM response:\n{content}")
        # Parse action and confidence
        action, confidence = self.parse_llm_response(content)
        if action not in ('buy', 'sell', 'hold'):
            logger.error(f"[GaiaLLMStrategy] Invalid action from LLM: {action}")
            return None
        if action == 'hold':
            return None
        # Use a default quantity; risk manager will adjust
        signal_type = SignalType.BUY if action == 'buy' else SignalType.SELL
        signal = StrategySignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=market_data['ohlcv']['close'].iloc[-1],
            quantity=0.1,
            timestamp=market_data.get('timestamp'),
            strategy_name='gaia_llm',
            metadata={
                'llm_prompt': prompt,
                'llm_response': content,
                'market_data': market_data
            }
        )
        return signal

    def parse_llm_response(self, content: str) -> (str, float):
        """Parse LLM response for action and confidence."""
        action = 'hold'
        confidence = 0.5
        for line in content.splitlines():
            if line.lower().startswith('action:'):
                val = line.split(':', 1)[1].strip().lower()
                if val in ('buy', 'sell', 'hold'):
                    action = val
            if line.lower().startswith('confidence:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except Exception:
                    confidence = 0.5
        return action, confidence 