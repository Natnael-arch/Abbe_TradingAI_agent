#!/usr/bin/env python3
"""
Backtesting Script for Trading AI Agent

This script tests the trading strategies on historical data from Binance.
It simulates trading on past data to evaluate strategy performance.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from loguru import logger

from src.utils.config import Config
from src.data.market_data import MarketDataProvider
from src.strategies.strategy_manager import StrategyManager
from src.risk.risk_manager import RiskManager
from src.performance.performance_tracker import PerformanceTracker


class Backtester:
    """
    Backtesting engine for trading strategies.
    
    This class:
    - Fetches historical data from Binance
    - Simulates trading with real market conditions
    - Evaluates strategy performance
    - Generates detailed performance reports
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.market_data = MarketDataProvider(config)
        self.strategy_manager = StrategyManager(config)
        self.risk_manager = RiskManager(config)
        self.performance_tracker = PerformanceTracker(config)
        
        # Backtesting state
        self.initial_balance = 10000  # $10,000 starting balance
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        logger.info("Backtester initialized")
    
    async def run_backtest(self, start_date: str, end_date: str, symbols: List[str] = None):
        """
        Run backtest on historical data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of symbols to backtest (default: config symbols)
        """
        if symbols is None:
            symbols = self.config.trading.symbols
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")
        
        try:
            # Connect to exchange
            await self.market_data.connect()
            
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
        finally:
            await self.market_data.disconnect()
    
    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical OHLCV data for a symbol."""
        try:
            # Convert dates flexibly
            def parse_date(date_str):
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Date {date_str} is not in a recognized format")
            start_dt = parse_date(start_date)
            end_dt = parse_date(end_date)
            # Calculate days difference
            days = (end_dt - start_dt).days
            # Fetch historical data
            if self.market_data.exchange is not None:
                ohlcv = await self.market_data.get_historical_data(symbol, days)
                return ohlcv
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
            if i % 100 == 0:  # Progress update every 100 iterations
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
        symbol = signal.symbol  # Fix: define symbol
        
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
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': current_price,
                    'cost': cost,
                    'balance': self.current_balance
                })
                logger.info(f"BUY {quantity} {symbol} @ ${current_price:.2f}")
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
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'price': current_price,
                    'revenue': revenue,
                    'pnl': pnl,
                    'balance': self.current_balance
                })
                logger.info(f"SELL {sell_quantity} {symbol} @ ${current_price:.2f} (P&L: ${pnl:.2f})")
    
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
        
        # Force-close all open positions at the latest price
        if self.positions:
            logger.info("Force-closing all open positions at the latest price...")
            # Use the last known price from the equity curve or trades
            last_price = None
            if self.trades:
                last_price = self.trades[-1]['price']
            for symbol, position in list(self.positions.items()):
                close_price = last_price if last_price is not None else position['entry_price']
                sell_quantity = position['quantity']
                revenue = sell_quantity * close_price
                pnl = (close_price - position['entry_price']) * sell_quantity
                self.current_balance += revenue
                self.trades.append({
                    'timestamp': 'FORCE_CLOSE',
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': sell_quantity,
                    'price': close_price,
                    'revenue': revenue,
                    'pnl': pnl,
                    'balance': self.current_balance
                })
                logger.info(f"FORCE SELL {sell_quantity} {symbol} @ ${close_price:.2f} (P&L: ${pnl:.2f})")
                del self.positions[symbol]
        
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
        final_balance = self.current_balance
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
        
        logger.info("="*50)


async def main():
    """Main backtesting function."""
    # Load configuration
    config = Config()
    
    # Create backtester
    backtester = Backtester(config)
    
    # Run backtest for one day: from yesterday to today
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    await backtester.run_backtest(start_date, end_date)


if __name__ == "__main__":
    asyncio.run(main()) 