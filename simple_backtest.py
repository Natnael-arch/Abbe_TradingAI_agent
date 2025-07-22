#!/usr/bin/env python3
"""
Simple Backtesting Script

This script tests the trading strategies on historical data from Binance
without circular import issues.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from loguru import logger
import ccxt

from src.utils.config import Config
from src.strategies.strategy_manager import StrategyManager
from src.risk.risk_manager import RiskManager


class SimpleBacktester:
    """
    Simple backtesting engine for trading strategies.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.strategy_manager = StrategyManager(config)
        self.risk_manager = RiskManager(config)
        
        # Backtesting state
        self.initial_balance = 10000  # $10,000 starting balance
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Setup exchange
        self.exchange = None
        self._setup_exchange()
        
        logger.info("Simple Backtester initialized")
    
    def _setup_exchange(self):
        """Setup exchange connection."""
        try:
            exchange_config = self.config.get_exchange_config()
            
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_config.name)
            self.exchange = exchange_class({
                'apiKey': exchange_config.api_key,
                'secret': exchange_config.secret,
                'sandbox': exchange_config.sandbox,
                'testnet': exchange_config.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            logger.info(f"Exchange {exchange_config.name} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            self.exchange = None
    
    async def run_backtest(self, start_date: str, end_date: str, symbols: List[str] = None):
        """
        Run backtest on historical data.
        """
        if symbols is None:
            symbols = self.config.trading.symbols
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")
        
        try:
            # Get historical data for all symbols
            historical_data = {}
            for symbol in symbols:
                logger.info(f"Fetching historical data for {symbol}...")
                data = await self._get_historical_data(symbol, start_date, end_date)
                if data is not None:
                    historical_data[symbol] = data
                    logger.info(f"Loaded {len(data)} data points for {symbol}")
            
            if not historical_data:
                logger.error("No historical data available")
                return
            
            # Run simulation
            await self._run_simulation(historical_data)
            
            # Generate performance report
            self._generate_performance_report()
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
    
    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical OHLCV data for a symbol."""
        try:
            # Convert dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Calculate days difference
            days = (end_dt - start_dt).days
            
            if self.exchange is not None:
                # Fetch real historical data
                ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=days * 24 * 4)
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                # Generate mock historical data
                return self._generate_mock_historical_data(symbol, start_dt, end_dt)
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _generate_mock_historical_data(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Generate realistic mock historical data."""
        # Generate date range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='1H')
        
        # Generate realistic price data
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
        volatility = 0.02  # 2% daily volatility
        
        prices = []
        current_price = base_price
        
        for _ in range(len(date_range)):
            # Random walk with volatility
            change = np.random.normal(0, volatility / np.sqrt(24))  # Hourly volatility
            current_price *= (1 + change)
            prices.append(current_price)
        
        # Create OHLCV data
        ohlcv_data = []
        for i, (ts, price) in enumerate(zip(date_range, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price if i == 0 else prices[i-1]
            volume = np.random.uniform(1000, 10000)
            
            ohlcv_data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(ohlcv_data)
        df.set_index('timestamp', inplace=True)
        return df
    
    async def _run_simulation(self, historical_data: Dict[str, pd.DataFrame]):
        """Run the trading simulation."""
        logger.info("Starting trading simulation...")
        
        # Get all timestamps
        all_timestamps = set()
        for symbol, data in historical_data.items():
            all_timestamps.update(data.index)
        
        timestamps = sorted(list(all_timestamps))
        
        # Process each timestamp
        for i, timestamp in enumerate(timestamps):
            if i % 50 == 0:  # Progress update every 50 iterations
                logger.info(f"Processing {i+1}/{len(timestamps)} timestamps...")
            
            # Prepare market data for current timestamp
            market_data = {}
            for symbol, data in historical_data.items():
                if timestamp in data.index:
                    # Get data up to current timestamp
                    current_data = data.loc[:timestamp].tail(100)  # Last 100 data points
                    
                    if len(current_data) >= 50:  # Need minimum data for indicators
                        market_data[symbol] = {
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'ohlcv': current_data,
                            'current_price': current_data.iloc[-1]['close'],
                            'volume_24h': current_data['volume'].sum(),
                            'price_change_24h': ((current_data.iloc[-1]['close'] - current_data.iloc[0]['close']) / current_data.iloc[0]['close']) * 100,
                            'high_24h': current_data['high'].max(),
                            'low_24h': current_data['low'].min()
                        }
            
            # Generate signals for each symbol
            for symbol, data in market_data.items():
                await self._process_symbol(symbol, data, timestamp)
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': self.current_balance,
                'positions_value': self._calculate_positions_value(market_data)
            })
    
    async def _process_symbol(self, symbol: str, market_data: Dict, timestamp: datetime):
        """Process trading signals for a symbol."""
        try:
            # Generate signals
            signals = await self.strategy_manager.generate_signals(symbol, market_data)
            
            if signals:
                # Combine signals
                final_signal = await self.strategy_manager.combine_signals(signals)
                
                if final_signal and final_signal.action != 'hold':
                    # Apply risk management
                    risk_adjusted_signal = await self.risk_manager.apply_risk_management(
                        symbol, final_signal, market_data
                    )
                    
                    if risk_adjusted_signal:
                        # Execute trade
                        await self._execute_trade(risk_adjusted_signal, timestamp)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def _execute_trade(self, signal, timestamp: datetime):
        """Execute a trade in the backtest."""
        current_price = signal.price
        quantity = signal.quantity
        symbol = signal.symbol
        
        if signal.action == 'buy':
            cost = quantity * current_price
            if cost <= self.current_balance:
                # Open position
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': timestamp
                    }
                else:
                    # Add to existing position
                    total_quantity = self.positions[symbol]['quantity'] + quantity
                    total_cost = (self.positions[symbol]['quantity'] * self.positions[symbol]['entry_price']) + cost
                    self.positions[symbol]['entry_price'] = total_cost / total_quantity
                    self.positions[symbol]['quantity'] = total_quantity
                
                self.current_balance -= cost
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': signal.symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': current_price,
                    'cost': cost,
                    'balance': self.current_balance
                })
                
                logger.info(f"BUY {quantity} {signal.symbol} @ ${current_price:.2f}")
        
        elif signal.action == 'sell':
            if symbol in self.positions:
                position = self.positions[symbol]
                sell_quantity = min(quantity, position['quantity'])
                revenue = sell_quantity * current_price
                
                # Update position
                position['quantity'] -= sell_quantity
                if position['quantity'] <= 0:
                    del self.positions[symbol]
                
                self.current_balance += revenue
                
                # Calculate P&L
                pnl = (current_price - position['entry_price']) * sell_quantity
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': signal.symbol,
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'price': current_price,
                    'revenue': revenue,
                    'pnl': pnl,
                    'balance': self.current_balance
                })
                
                logger.info(f"SELL {sell_quantity} {signal.symbol} @ ${current_price:.2f} (P&L: ${pnl:.2f})")
    
    def _calculate_positions_value(self, market_data: Dict) -> float:
        """Calculate current value of all positions."""
        total_value = 0
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['current_price']
                total_value += position['quantity'] * current_price
        return total_value
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        logger.info("\n" + "="*50)
        logger.info("BACKTEST PERFORMANCE REPORT")
        logger.info("="*50)
        
        # Calculate metrics
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        
        # P&L analysis
        total_pnl = sum([t.get('pnl', 0) for t in sell_trades])
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        
        # Final balance
        final_balance = self.current_balance + self._calculate_positions_value({})
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Print results
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Balance: ${final_balance:,.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Total P&L: ${total_pnl:,.2f}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Buy Trades: {len(buy_trades)}")
        logger.info(f"Sell Trades: {len(sell_trades)}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        
        if winning_trades:
            avg_win = sum([t.get('pnl', 0) for t in winning_trades]) / len(winning_trades)
            logger.info(f"Average Win: ${avg_win:.2f}")
        
        if losing_trades:
            avg_loss = sum([t.get('pnl', 0) for t in losing_trades]) / len(losing_trades)
            logger.info(f"Average Loss: ${avg_loss:.2f}")
        
        # Current positions
        if self.positions:
            logger.info(f"\nCurrent Positions:")
            for symbol, position in self.positions.items():
                logger.info(f"  {symbol}: {position['quantity']} @ ${position['entry_price']:.2f}")
        
        logger.info("="*50)


async def main():
    """Main backtesting function."""
    # Load configuration
    config = Config()
    
    # Create backtester
    backtester = SimpleBacktester(config)
    
    # Run backtest for last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    await backtester.run_backtest(start_date, end_date)


if __name__ == "__main__":
    asyncio.run(main()) 