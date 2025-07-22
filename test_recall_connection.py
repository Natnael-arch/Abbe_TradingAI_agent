import asyncio
from src.utils.config import Config
from src.competition.recall_client import RecallClient

async def main():
    config = Config("config/config.yaml")
    recall_config = config.get_recall_config()
    async with RecallClient(recall_config) as client:
        success = await client.test_connection()
        if success:
            print("Recall API connection successful.")
        else:
            print("Recall API connection failed.")

if __name__ == "__main__":
    asyncio.run(main()) 