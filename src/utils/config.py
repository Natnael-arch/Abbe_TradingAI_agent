"""
Configuration management for the trading agent.

This module handles loading and validation of configuration files,
including trading parameters, risk management settings, and API credentials.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger


@dataclass
class ExchangeConfig:
    """Exchange configuration settings."""
    name: str
    testnet: bool
    api_key: str
    secret: str
    sandbox: bool


@dataclass
class TradingConfig:
    """Trading parameters configuration."""
    symbols: List[str]
    base_currency: str
    position_size: float
    max_positions: int
    min_order_size: float


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'weight': self.weight,
            'parameters': self.parameters or {}
        }


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown: float
    stop_loss: float
    take_profit: float
    trailing_stop: bool
    max_daily_trades: int
    volatility_threshold: float
    min_confidence: float = 0.3
    min_order_size: float = 10.0
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_position_size: float = 1000.0


@dataclass
class PerformanceConfig:
    """Performance tracking configuration."""
    track_metrics: bool
    save_trades: bool
    log_level: str
    metrics_interval: int


@dataclass
class DataConfig:
    """Data configuration settings."""
    timeframe: str
    lookback_period: int
    update_interval: int


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    mode: str
    gaia_api_key: str = ""
    gaia_endpoint: str = ""
    auto_restart: bool = True
    health_check_interval: int = 300


@dataclass
class CompetitionConfig:
    """Competition settings."""
    agent_id: str
    competition_mode: bool
    fair_start: bool
    performance_reporting: bool


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    file: str
    max_size: str
    backup_count: int


@dataclass
class RecallConfig:
    """Recall competition API configuration."""
    api_key: str = ""


@dataclass
class GaiaConfig:
    enabled: bool = True
    api_key: str = ""
    base_url: str = ""


class Config:
    """Main configuration class that loads and manages all settings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Parse configuration sections
        self.exchange = ExchangeConfig(**config_data.get('exchange', {}))
        self.trading = TradingConfig(**config_data.get('trading', {}))
        self.strategy = StrategyConfig(**config_data.get('strategy', {}))
        self.risk = RiskConfig(**config_data.get('risk', {}))
        self.performance = PerformanceConfig(**config_data.get('performance', {}))
        self.data = DataConfig(**config_data.get('data', {}))
        self.deployment = DeploymentConfig(**config_data.get('deployment', {}))
        self.competition = CompetitionConfig(**config_data.get('competition', {}))
        self.logging = LoggingConfig(**config_data.get('logging', {}))
        self.recall = RecallConfig(**config_data.get('recall', {}))
        self.gaia = GaiaConfig(**config_data.get('gaia', {}))
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    def get_exchange_config(self) -> ExchangeConfig:
        """Get exchange configuration."""
        return self.exchange
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self.trading
    
    def get_strategy_config(self) -> StrategyConfig:
        """Get strategy configuration."""
        return self.strategy
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk management configuration."""
        return self.risk
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance tracking configuration."""
        return self.performance
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.data
    
    def get_deployment_config(self) -> DeploymentConfig:
        """Get deployment configuration."""
        return self.deployment
    
    def get_competition_config(self) -> CompetitionConfig:
        """Get competition configuration."""
        return self.competition
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.logging
    
    def get_recall_config(self) -> RecallConfig:
        """Get Recall competition configuration."""
        return self.recall
    
    def get_gaia_config(self) -> GaiaConfig:
        return self.gaia
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate exchange settings
            if not self.exchange.name:
                raise ValueError("Exchange name is required")
            
            # Validate trading settings
            if not self.trading.symbols:
                raise ValueError("At least one trading symbol is required")
            
            if self.trading.position_size <= 0 or self.trading.position_size > 1:
                raise ValueError("Position size must be between 0 and 1")
            
            # Validate risk settings
            if self.risk.max_drawdown <= 0 or self.risk.max_drawdown > 1:
                raise ValueError("Max drawdown must be between 0 and 1")
            
            if self.risk.stop_loss <= 0:
                raise ValueError("Stop loss must be positive")
            
            # Validate data settings
            valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            if self.data.timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Must be one of: {valid_timeframes}")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'exchange': self.exchange.__dict__,
            'trading': self.trading.__dict__,
            'strategy': self.strategy.__dict__,
            'risk': self.risk.__dict__,
            'performance': self.performance.__dict__,
            'data': self.data.__dict__,
            'deployment': self.deployment.__dict__,
            'competition': self.competition.__dict__,
            'logging': self.logging.__dict__
        } 