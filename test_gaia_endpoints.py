#!/usr/bin/env python3
"""
Test Gaia API endpoints and provide fallback options
"""

import asyncio
import aiohttp
from src.utils.config import Config

async def test_gaia_endpoint(base_url: str, api_key: str) -> bool:
    """Test a Gaia endpoint."""
    url = f"{base_url}/models"  # Use /models endpoint instead of /status
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, timeout=10) as resp:
                print(f"  {base_url}: {resp.status}")
                if resp.status == 200:
                    data = await resp.json()
                    print(f"    Models available: {[m['id'] for m in data.get('data', [])]}")
                return resp.status == 200
    except Exception as e:
        print(f"  {base_url}: Error - {e}")
        return False

async def main():
    config = Config("config/config.yaml")
    gaia_config = config.get_gaia_config()
    api_key = gaia_config.api_key
    
    # Extract potential node ID from API key
    node_id_from_key = api_key.split('-')[1] if '-' in api_key else api_key[:8]
    
    # Test different possible URLs based on Gaia documentation
    test_urls = [
        # Try with extracted node ID
        f"https://{node_id_from_key}.gaia.domains/v1",
        # Try with full API key as node ID
        f"https://{api_key}.gaia.domains/v1",
        # Try common node naming patterns
        f"https://node1.gaia.domains/v1",
        f"https://gaia-node.gaia.domains/v1",
        f"https://trading-node.gaia.domains/v1",
        f"https://public-node.gaia.domains/v1",
        f"https://demo-node.gaia.domains/v1",
        # Try without subdomain
        f"https://gaia.domains/v1",
        f"https://api.gaia.domains/v1",
    ]
    
    print("Testing Gaia endpoints:")
    print(f"API Key: {api_key}")
    print(f"Extracted node ID: {node_id_from_key}")
    print()
    
    working_endpoints = []
    for url in test_urls:
        success = await test_gaia_endpoint(url, api_key)
        if success:
            working_endpoints.append(url)
    
    print(f"\nWorking endpoints: {working_endpoints}")
    if working_endpoints:
        print(f"✅ Found {len(working_endpoints)} working endpoint(s)")
        print(f"Update your config.yaml with: base_url: \"{working_endpoints[0]}\"")
    else:
        print("❌ No working endpoints found")
        print("You need to get the correct node ID from your Gaia node operator")

if __name__ == "__main__":
    asyncio.run(main()) 