"""
AI Conversational System 
Codename: Cloudy

This script implements an AI conversational system utilizing OpenAI's GPT-3.5 language model.
The system interacts with users through speech recognition, AI responses, and facial recognition.

Author: Sachu-Settan i.e, Sanooj
Lisensed under GNU GPLv3 License.
"""

import aiohttp
import asyncio
from main import logger,CHECK_INTERNET_URL
class NoInternetConnection(Exception):
    pass

async def check_internet():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(CHECK_INTERNET_URL) as response:
                if response.status == 200:
                    logger.info("Internet is connected")
                    return True
                else:
                    raise NoInternetConnection("Internet is not connected")
    except aiohttp.ClientError:
        raise NoInternetConnection("No internet connection available")
    except asyncio.CancelledError:
        logger.exception("Task cancelled by user.")
        raise
    except KeyboardInterrupt:
        logger.exception("Conversation terminated by user.")
        raise

async def check_internet_connection():
    try:
        connected = await check_internet()
        return connected
    except NoInternetConnection as e:
        logger.exception(f"Internet check failed: {e}")
        return False
