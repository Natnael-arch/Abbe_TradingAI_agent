"""
Strategy types and data structures.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class StrategySignal:
    """Represents a trading signal from a strategy."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    quantity: float
    timestamp: float
    strategy_name: str
    metadata: Dict[str, Any]
    
    @property
    def action(self) -> str:
        """Get the action string."""
        return self.signal_type.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'price': self.price,
            'quantity': self.quantity,
            'timestamp': self.timestamp,
            'strategy_name': self.strategy_name,
            'metadata': self.metadata
        }


@dataclass
class TradeSignal:
    """Represents a trading signal with all necessary information."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    price: float
    quantity: float
    timestamp: float
    strategy: str
    metadata: Dict[str, Any]


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {} 