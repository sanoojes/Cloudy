import aiohttp
import asyncio
from main import logger,CHECK_INTERNET_URL

# ----------------------------------------------------------------------------
class NoInternetConnection(Exception):
    pass
# ----------------------------------------------------------------------------
async def check_internet():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(CHECK_INTERNET_URL) as response:
                if response.status == 200:
                    logger.info("Internet is connected")
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
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def check_internet_connection():
    try:
        await check_internet()
    except NoInternetConnection as e:
        logger.exception(f"Internet check failed: {e}")
# ----------------------------------------------------------------------------