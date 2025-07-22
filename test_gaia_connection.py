#!/usr/bin/env python3
"""
Test Gaia connection before running backtest.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.ai.gaia_client import GaiaClient

async def test_gaia_connection():
    """Test if Gaia connection is working."""
    
    print("Testing Gaia Connection...")
    print("=" * 40)
    
    try:
        # Load config
        config = Config("config/config.yaml")
        gaia_config = config.get_gaia_config()
        
        print(f"Endpoint: {gaia_config.base_url}")
        print(f"API Key: {gaia_config.api_key[:20]}...")
        
        # Test connection
        async with GaiaClient(gaia_config) as client:
            # Test basic inference
            test_prompt = "You are a trading AI. Respond with: Action: buy\nConfidence: 0.8\nReason: Test response"
            
            print("\nSending test prompt to Gaia...")
            response = await client.infer(test_prompt)
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                print(f"✅ Gaia Response: {content}")
                print("✅ Connection successful!")
                return True
            else:
                print("❌ No valid response from Gaia")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def main():
    """Run the connection test."""
    success = await test_gaia_connection()
    
    if success:
        print("\n🎉 Gaia connection is working! You can now run the backtest.")
        print("Run: python backtest_gaia_strategy.py")
    else:
        print("\n⚠️ Gaia connection failed. Consider using the competition strategy instead.")
        print("Update config.yaml to use 'competition' strategy instead of 'gaia_llm'")

if __name__ == "__main__":
    asyncio.run(main()) 