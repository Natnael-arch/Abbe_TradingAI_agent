# Abbe Crypto Trading AI Agent

A modular, autonomous AI trading agent designed for competitive hackathons where agents compete based on trading performance.

## Architecture Overview

### Core Components
- **Data Layer**: Real-time and historical crypto market data via CCXT
- **Strategy Layer**: RSI/MACD indicators and ML-based trading strategies
- **Decision Engine**: Autonomous buy/sell/hold decisions
- **Risk Management**: Position sizing, stop-loss, and portfolio management
- **Performance Tracking**: P&L, Sharpe ratio, and competition metrics
- **Deployment**: Gaia hosting for inference, Mastra + MCP for agent registration

### Technology Stack
- **Trading Engine**: CCXT for exchange connectivity
- **Data Analysis**: pandas, numpy, ta-lib for technical indicators
- **ML/AI**: scikit-learn, tensorflow/pytorch for advanced strategies
- **Deployment**: Gaia for hosting, Mastra + MCP for agent management
- **Security**: Lit Protocol for secure signing and access control
- **Monitoring**: Real-time performance tracking and logging

## Project Structure
```
tradingAI/
├── src/
│   ├── agent/           # Core agent logic
│   ├── strategies/      # Trading strategies
│   ├── data/           # Market data handling
│   ├── risk/           # Risk management
│   ├── performance/    # Performance tracking
│   └── deployment/     # Deployment configurations
├── config/             # Configuration files
├── tests/              # Unit and integration tests
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   cp config/config.example.yaml config/config.yaml
   # Edit config.yaml with your API keys
   ```

3. **Run the Agent**
   ```bash
   python src/main.py
   ```

## Features

### Trading Strategies
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **ML Strategies**: LSTM price prediction, reinforcement learning
- **Hybrid Approaches**: Combine multiple signals for robust decisions

### Hybrid + Gaia LLM Ensemble Strategy

This project features an advanced ensemble approach that combines traditional technical analysis with AI-powered contextual reasoning for superior trading decisions:

#### 1. Hybrid Strategy (Technical Analysis)
- The **HybridStrategy** leverages multiple technical indicators:
  - **RSI (Relative Strength Index):** Detects overbought/oversold conditions.
  - **MACD (Moving Average Convergence Divergence):** Identifies trend direction and momentum.
  - **Bollinger Bands:** Measures volatility and price extremes.
  - **Volume Analysis:** Confirms signal strength.
  - **Multi-timeframe Confirmation:** Increases robustness by checking signals across timeframes.
- The hybrid logic is aggressive for competitions: only one indicator needs to agree for a trade, and confidence thresholds are tuned for more frequent signals.

#### 2. Gaia LLM Strategy (Context & Pattern Recognition)
- The **GaiaLLMStrategy** uses a Large Language Model (LLM) node (via Gaia) to:
  - Analyze structured market data (price, RSI, MACD, Bollinger Bands, support/resistance, trend, etc.).
  - Understand market context, patterns, and narrative beyond what technicals alone can capture.
  - Generate a natural-language rationale and a confidence score for each action (buy/sell/hold).
- The LLM is prompted with up-to-date technicals and market context, and returns a structured response parsed by the agent.

#### 3. StrategyManager (Ensembling)
- The **StrategyManager** orchestrates both strategies:
  - Runs both the Hybrid and Gaia LLM strategies in parallel on the same market data.
  - Collects their signals and confidence scores.
  - Combines/ensembles the results using a weighted voting or rule-based system (configurable).
  - The final trading action is chosen based on the ensemble, leveraging both technical rigor and AI-driven context.
- This approach ensures the agent can:
  - React quickly to technical signals
  - Adapt to changing market regimes
  - Incorporate broader context and pattern recognition from the LLM

**This ensemble design provides a unique edge in both live trading and competitive environments, balancing the strengths of algorithmic and AI-driven decision making.**

### Risk Management
- Position sizing based on volatility
- Dynamic stop-loss and take-profit
- Portfolio diversification
- Maximum drawdown protection

### Performance Tracking
- Real-time P&L monitoring
- Sharpe ratio and other metrics
- Competition ranking system
- Detailed trade logging

### Deployment
- Gaia hosting for continuous operation
- Mastra + MCP for agent registration
- Lit Protocol for secure access control
- Docker containerization

## Competition Features

- **Autonomous Operation**: Runs continuously without human intervention
- **Fair Competition**: Isolated environments with equal starting conditions
- **Performance Metrics**: Profit/loss, risk-adjusted returns, consistency
- **Intelligent Behavior**: Adaptive strategies based on market conditions



## License

MIT License - See LICENSE file for details 