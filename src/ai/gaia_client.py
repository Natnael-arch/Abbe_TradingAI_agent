import aiohttp
from loguru import logger
from typing import Dict, Any, Optional
import traceback
from src.utils.config import GaiaConfig

class GaiaClient:
    def __init__(self, config: GaiaConfig):
        self.api_key = config.api_key
        self.base_url = config.base_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_connection(self) -> bool:
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with self.session.post(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m['id'] for m in data.get('data', [])]
                    logger.info(f"Gaia node connection successful. Available models: {models}")
                    return True
                logger.error(f"Gaia node connection failed: {resp.status}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Gaia node: {type(e).__name__} - {e}")
            logger.debug(traceback.format_exc()) # Add traceback for debugging
            return False

    async def infer(self, prompt: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "trading-ai-v1",
            "messages": [{"role": "user", "content": prompt}],
        }
        if params:
            payload.update(params)
        try:
            async with self.session.post(url, headers=headers, json=payload, timeout=30) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"Gaia inference failed: {resp.status}")
                return None
        except Exception as e:
            logger.error(f"Error during Gaia inference: {type(e).__name__} - {e}")
            logger.debug(traceback.format_exc()) # Add traceback for debugging
            return None 