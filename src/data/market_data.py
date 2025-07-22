"""
Market data provider for real-time crypto trading data.
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

from ..utils.config import Config


class MarketDataProvider:
    """
    Handles real-time and historical market data from crypto exchanges.
    
    Uses CCXT library to connect to exchanges and fetch:
    - OHLCV data (Open, High, Low, Close, Volume)
    - Order book data
    - Ticker information
    - Account balances and positions
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.exchange = None
        self.is_connected = False
        self.last_data = {}
        self.data_cache = {}
        
        # Initialize exchange connection
        self._setup_exchange()
    
    def _setup_exchange(self):
        """Initialize the exchange connection."""
        exchange_config = self.config.get_exchange_config()
        
        # Check if API keys are provided
        if exchange_config.api_key == "YOUR_API_KEY" or exchange_config.secret == "YOUR_SECRET_KEY":
            logger.warning("API keys not configured, using mock data")
            self.exchange = None
            return
        
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_config.name)
            
            # Configure exchange settings
            exchange_config_dict = {
                'apiKey': exchange_config.api_key,
                'secret': exchange_config.secret,
                'sandbox': exchange_config.sandbox,
                'testnet': exchange_config.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Explicitly use spot market
                }
            }
            
            # Add passphrase for Coinbase Pro
            if exchange_config.name == 'coinbasepro' and hasattr(exchange_config, 'passphrase'):
                exchange_config_dict['passphrase'] = exchange_config.passphrase
            
            self.exchange = exchange_class(exchange_config_dict)
            
            logger.info(f"Exchange {exchange_config.name} initialized with real connection")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            logger.info("Falling back to mock data")
            self.exchange = None
    
    async def connect(self):
        """Connect to the exchange."""
        if self.exchange is None:
            logger.error("Exchange not initialized.")
            return
        try:
            # Use the synchronous version of load_markets
            self.exchange.load_markets()
            self.is_connected = True
            logger.info("Successfully connected to exchange")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            self.is_connected = False
            raise ConnectionError(f"Could not connect to the exchange: {e}") from e
    
    async def disconnect(self):
        """Disconnect from the exchange."""
        self.is_connected = False
        logger.info("Disconnected from exchange")
    
    async def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest market data for a symbol.
        
        Returns:
            Dict containing OHLCV data, ticker, and order book
        """
        if not self.is_connected:
            logger.warning("Not connected to exchange")
            return None
        
        try:
            # Check if we have real exchange access
            if self.exchange is not None and hasattr(self.exchange, 'fetch_ohlcv'):
                timeframe = self.config.get_data_config().timeframe
                
                # Fetch OHLCV data (sync version)
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    limit=self.config.get_data_config().lookback_period
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Fetch latest ticker and order book data
                ticker = self.exchange.fetch_ticker(symbol)
                orderbook = self.exchange.fetch_order_book(symbol, limit=10)
                
                # Prepare market data
                market_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'ohlcv': df,
                    'ticker': ticker,
                    'orderbook': orderbook,
                    'current_price': ticker['last'],
                    'volume_24h': ticker['quoteVolume'],
                    'price_change_24h': ticker['percentage'],
                    'high_24h': ticker['high'],
                    'low_24h': ticker['low']
                }
            else:
                # Use mock data
                market_data = self._generate_mock_data(symbol)
            
            # Cache the data
            self.data_cache[symbol] = market_data
            self.last_data[symbol] = datetime.now()
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            # Fallback to mock data
            logger.info(f"Using mock data for {symbol}")
            return self._generate_mock_data(symbol)
    
    def _generate_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock market data for demonstration."""
        import numpy as np
        import random
        
        # Generate realistic mock data
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        volatility = 0.02  # 2% daily volatility
        
        # Generate OHLCV data
        timestamps = pd.date_range(end=datetime.now(), periods=100, freq='h')
        prices = []
        current_price = base_price
        
        for _ in range(100):
            # Random walk with volatility
            change = np.random.normal(0, volatility / np.sqrt(24))  # Hourly volatility
            current_price *= (1 + change)
            prices.append(current_price)
        
        # Create OHLCV data
        ohlcv_data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price if i == 0 else prices[i-1]
            volume = random.uniform(1000, 10000)
            
            ohlcv_data.append([
                int(ts.timestamp() * 1000),  # timestamp in ms
                open_price,
                high,
                low,
                price,
                volume
            ])
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Mock ticker data
        current_price = prices[-1]
        ticker = {
            'last': current_price,
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'high': max(prices),
            'low': min(prices),
            'volume': sum([row[5] for row in ohlcv_data]),
            'quoteVolume': sum([row[5] * row[4] for row in ohlcv_data]),
            'percentage': ((current_price - prices[0]) / prices[0]) * 100
        }
        
        # Mock order book
        orderbook = {
            'bids': [[current_price * 0.999, random.uniform(1, 10)] for _ in range(10)],
            'asks': [[current_price * 1.001, random.uniform(1, 10)] for _ in range(10)]
        }
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'ohlcv': df,
            'ticker': ticker,
            'orderbook': orderbook,
            'current_price': current_price,
            'volume_24h': ticker['quoteVolume'],
            'price_change_24h': ticker['percentage'],
            'high_24h': ticker['high'],
            'low_24h': ticker['low']
        }
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for backtesting.
        
        Args:
            symbol: Trading symbol
            days: Number of days to fetch
            
        Returns:
            DataFrame with historical OHLCV data
        """
        if not self.is_connected:
            return None
        
        try:
            timeframe = self.config.get_data_config().timeframe
            
            # Calculate limit based on timeframe and days
            if timeframe == '1m':
                limit = days * 24 * 60
            elif timeframe == '5m':
                limit = days * 24 * 12
            elif timeframe == '15m':
                limit = days * 24 * 4
            elif timeframe == '1h':
                limit = days * 24
            elif timeframe == '4h':
                limit = days * 6
            elif timeframe == '1d':
                limit = days
            else:
                limit = days * 24  # Default to hourly
            
            # Fetch historical data
            if self.exchange is not None:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get current account balance."""
        if not self.is_connected:
            return {}
        
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total': balance.get('total', {}),
                'free': balance.get('free', {}),
                'used': balance.get('used', {})
            }
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return {}
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions."""
        if not self.is_connected:
            return []
        
        try:
            positions = self.exchange.fetch_positions()
            return [
                {
                    'symbol': pos['symbol'],
                    'side': pos['side'],
                    'size': pos['size'],
                    'notional': pos['notional'],
                    'unrealized_pnl': pos['unrealizedPnl'],
                    'entry_price': pos['entryPrice'],
                    'mark_price': pos['markPrice']
                }
                for pos in positions if pos['size'] != 0
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def execute_trade(self, signal) -> Optional[Dict[str, Any]]:
        """
        Execute a trade based on a signal.
        
        Args:
            signal: TradeSignal object with trade details
            
        Returns:
            Order information if successful, None otherwise
        """
        if not self.is_connected:
            logger.warning("Not connected to exchange")
            return None
        
        try:
            # Prepare order parameters
            order_params = {
                'symbol': signal.symbol,
                'type': 'market',  # Use market orders for immediate execution
                'side': signal.action,
                'amount': signal.quantity,
                'params': {}
            }
            
            # Execute the order
            if signal.action == 'buy':
                order = self.exchange.create_market_buy_order(
                    signal.symbol, signal.quantity
                )
            elif signal.action == 'sell':
                order = self.exchange.create_market_sell_order(
                    signal.symbol, signal.quantity
                )
            else:
                logger.warning(f"Invalid action: {signal.action}")
                return None
            
            logger.info(f"Order executed: {order['id']} for {signal.symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            return None
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific order."""
        if not self.is_connected:
            return None
        
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Error fetching order status: {e}")
            return None
    
    def get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data for a symbol."""
        return self.data_cache.get(symbol)
    
    def is_data_fresh(self, symbol: str, max_age_seconds: int = 60) -> bool:
        """Check if cached data is fresh enough."""
        if symbol not in self.last_data:
            return False
        
        age = (datetime.now() - self.last_data[symbol]).total_seconds()
        return age <= max_age_seconds
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available market data."""
        return {
            'connected': self.is_connected,
            'cached_symbols': list(self.data_cache.keys()),
            'last_updates': {
                symbol: self.last_data.get(symbol, 'Never').isoformat() 
                if isinstance(self.last_data.get(symbol), datetime) 
                else str(self.last_data.get(symbol, 'Never'))
                for symbol in self.data_cache.keys()
            }
        } 