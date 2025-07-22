import aiohttp
from typing import Optional, Dict, Any
from loguru import logger
from src.utils.config import RecallConfig

class RecallClient:
    """
    Client for Recall competition API integration.
    Handles authentication and communication with Recall endpoints.
    """
    def __init__(self, config: RecallConfig):
        self.api_key = config.api_key
        self.base_url = "https://api.competitions.recall.network"  # Updated endpoint per Recall team
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_connection(self) -> bool:
        """
        Test connection to Recall API (e.g., ping or health endpoint).
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}/health"  # Replace with actual Recall health endpoint if different
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    logger.info("Recall API connection successful.")
                    return True
                else:
                    logger.warning(f"Recall API connection failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to Recall API: {e}")
            return False

    # Add more methods for registration, results, leaderboard, etc. as needed 