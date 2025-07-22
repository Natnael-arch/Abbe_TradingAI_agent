from src.utils.config import Config
from src.agent.trading_agent import TradingAgent

# Load config and initialize agent
config = Config("config/config.yaml")
agent = TradingAgent(config)

# Print performance summary
summary = agent.get_performance_summary()
print("\n=== Performance Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# Export trades to CSV
csv_file = agent.performance_tracker.export_trades()
print(f"\nTrade history exported to: {csv_file}") 