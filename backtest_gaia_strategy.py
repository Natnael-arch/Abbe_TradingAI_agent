#!/usr/bin/env python3
"""
Backtesting system for Gaia LLM strategy.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.strategies.gaia_llm_strategy import GaiaLLMStrategy
from src.strategies.types import StrategySignal

class Backtester:
    """Backtesting system for trading strategies."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.strategy = GaiaLLMStrategy(self.config.get_strategy_config().to_dict())
        self.initial_balance = 10000  # $10,000 starting balance
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    async def run_backtest(self, symbol: str, start_date: str, end_date: str):
        """Run backtest for a given period."""
        print(f"Running backtest for {symbol} from {start_date} to {end_date}")
        print("=" * 60)
        
        # Load historical data
        historical_data = await self._load_historical_data(symbol, start_date, end_date)
        if historical_data is None:
            print("❌ Failed to load historical data")
            return
            
        print(f"✅ Loaded {len(historical_data)} data points")
        
        # Run simulation
        await self._simulate_trading(historical_data, symbol)
        
        # Calculate and display results
        self._calculate_performance()
        self._display_results()
        self._plot_results()
        
    async def _load_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical OHLCV data."""
        try:
            # For demo purposes, create synthetic data that mimics real market behavior
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Generate realistic price data
            days = (end_dt - start_dt).days
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='5min')
            
            # Create realistic price movements
            np.random.seed(42)  # For reproducible results
            base_price = 50000  # BTC starting price
            returns = np.random.normal(0, 0.02, len(timestamps))  # 2% daily volatility
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # Create OHLCV data
            data = []
            for i, (ts, price) in enumerate(zip(timestamps, prices)):
                # Add some realistic OHLC variation
                high = price * (1 + abs(np.random.normal(0, 0.005)))
                low = price * (1 - abs(np.random.normal(0, 0.005)))
                open_price = price * (1 + np.random.normal(0, 0.002))
                volume = np.random.randint(100, 1000)
                
                data.append({
                    'timestamp': ts,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None
    
    async def _simulate_trading(self, data: pd.DataFrame, symbol: str):
        """Simulate trading with the strategy."""
        print(f"Simulating trading for {len(data)} periods...")
        
        for i in range(50, len(data)):  # Start after enough data for indicators
            # Get current market data
            current_data = data.iloc[:i+1]
            
            # Prepare market data for strategy
            market_data = {
                'ohlcv': current_data,
                'timestamp': current_data.index[-1].timestamp(),
                'symbol': symbol
            }
            
            # Generate signal
            signal = await self.strategy.generate_signal(symbol, market_data)
            
            # Execute trades
            if signal:
                await self._execute_trade(signal, current_data.iloc[-1])
            
            # Update equity curve
            self._update_equity_curve(current_data.iloc[-1]['close'])
            
            # Progress indicator
            if i % 100 == 0:
                print(f"Processed {i}/{len(data)} periods...")
    
    async def _execute_trade(self, signal: StrategySignal, current_price_data: pd.Series):
        """Execute a trade based on the signal."""
        current_price = current_price_data['close']
        
        if signal.signal_type.value == 'buy':
            # Calculate position size (15% of balance)
            position_size = self.balance * 0.15
            quantity = position_size / current_price
            
            # Check if we have enough balance
            if position_size <= self.balance:
                # Execute buy
                self.positions[signal.symbol] = {
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': signal.timestamp
                }
                self.balance -= position_size
                
                self.trades.append({
                    'timestamp': signal.timestamp,
                    'action': 'buy',
                    'price': current_price,
                    'quantity': quantity,
                    'value': position_size,
                    'balance': self.balance
                })
                
                print(f"🟢 BUY: {quantity:.4f} {signal.symbol} @ ${current_price:,.2f}")
        
        elif signal.signal_type.value == 'sell':
            if signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                quantity = position['quantity']
                entry_price = position['entry_price']
                
                # Calculate P&L
                pnl = (current_price - entry_price) * quantity
                sale_value = quantity * current_price
                
                # Execute sell
                self.balance += sale_value
                del self.positions[signal.symbol]
                
                self.trades.append({
                    'timestamp': signal.timestamp,
                    'action': 'sell',
                    'price': current_price,
                    'quantity': quantity,
                    'value': sale_value,
                    'pnl': pnl,
                    'balance': self.balance
                })
                
                print(f"🔴 SELL: {quantity:.4f} {signal.symbol} @ ${current_price:,.2f} | P&L: ${pnl:,.2f}")
    
    def _update_equity_curve(self, current_price: float):
        """Update equity curve with current portfolio value."""
        portfolio_value = self.balance
        
        # Add value of open positions
        for symbol, position in self.positions.items():
            portfolio_value += position['quantity'] * current_price
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': portfolio_value,
            'balance': self.balance,
            'positions_value': portfolio_value - self.balance
        })
    
    def _calculate_performance(self):
        """Calculate performance metrics."""
        if not self.trades:
            print("❌ No trades executed during backtest")
            return
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        buy_trades = len([t for t in self.trades if t['action'] == 'buy'])
        sell_trades = len([t for t in self.trades if t['action'] == 'sell'])
        
        # Calculate P&L
        total_pnl = sum([t.get('pnl', 0) for t in self.trades])
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trades if t.get('pnl', 0) < 0])
        
        # Calculate returns
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.balance
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        
        # Calculate win rate
        win_rate = (winning_trades / sell_trades * 100) if sell_trades > 0 else 0
        
        # Calculate average trade
        avg_win = np.mean([t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
        
        # Store results
        self.results = {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': final_equity,
            'initial_balance': self.initial_balance
        }
    
    def _display_results(self):
        """Display backtest results."""
        if not hasattr(self, 'results'):
            return
            
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        
        r = self.results
        print(f"💰 Initial Balance: ${r['initial_balance']:,.2f}")
        print(f"💰 Final Equity: ${r['final_equity']:,.2f}")
        print(f"📈 Total Return: {r['total_return_pct']:.2f}%")
        print(f"💵 Total P&L: ${r['total_pnl']:,.2f}")
        print()
        print(f"📊 Trading Statistics:")
        print(f"   Total Trades: {r['total_trades']}")
        print(f"   Buy Trades: {r['buy_trades']}")
        print(f"   Sell Trades: {r['sell_trades']}")
        print(f"   Win Rate: {r['win_rate']:.1f}%")
        print(f"   Winning Trades: {r['winning_trades']}")
        print(f"   Losing Trades: {r['losing_trades']}")
        print()
        print(f"📈 Average Performance:")
        print(f"   Average Win: ${r['avg_win']:,.2f}")
        print(f"   Average Loss: ${r['avg_loss']:,.2f}")
        print(f"   Profit Factor: {abs(r['avg_win']/r['avg_loss']):.2f}" if r['avg_loss'] != 0 else "   Profit Factor: N/A")
        
        # Competition assessment
        print("\n🏆 COMPETITION ASSESSMENT:")
        if r['total_return_pct'] > 20:
            print("   🥇 EXCELLENT - High chance of winning!")
        elif r['total_return_pct'] > 10:
            print("   🥈 GOOD - Competitive performance")
        elif r['total_return_pct'] > 5:
            print("   🥉 DECENT - Moderate performance")
        else:
            print("   ⚠️ NEEDS IMPROVEMENT - Consider strategy adjustments")
    
    def _plot_results(self):
        """Plot equity curve and trade distribution."""
        if not self.equity_curve:
            return
            
        try:
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            equity_df = pd.DataFrame(self.equity_curve)
            ax1.plot(equity_df['timestamp'], equity_df['equity'], label='Portfolio Value', linewidth=2)
            ax1.axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
            ax1.set_title('Portfolio Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Trade distribution
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                if 'pnl' in trades_df.columns:
                    pnl_data = trades_df['pnl'].dropna()
                    if len(pnl_data) > 0:
                        ax2.hist(pnl_data, bins=20, alpha=0.7, color='green')
                        ax2.set_title('Trade P&L Distribution')
                        ax2.set_xlabel('P&L ($)')
                        ax2.set_ylabel('Frequency')
                        ax2.axvline(x=0, color='r', linestyle='--', label='Break Even')
                        ax2.legend()
                        ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Charts saved to 'backtest_results.png'")
            
        except Exception as e:
            print(f"Error creating plots: {e}")

async def main():
    """Run the backtest."""
    backtester = Backtester()
    
    # Run backtest for last 7 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    await backtester.run_backtest("BTC/USDT", start_date, end_date)

if __name__ == "__main__":
    asyncio.run(main()) 