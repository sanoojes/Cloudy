"""
AI Conversational System 
Codename: Cloudy

This script implements an AI conversational system utilizing OpenAI's GPT-3.5 language model.
The system interacts with users through speech recognition, AI responses, and facial recognition.

Author: Sachu-Settan i.e, Sanooj
Lisensed under GNU GPLv3 License.
"""

import subprocess
import asyncio
from main import EDGE_VOICE_NAME, TTS_FILE, logger
# ----------------------------------------------------------------------------
async def gtts(text: str):
    try:
        from gtts import gTTS
        logger.debug("Trying to convert text to speech using gTTS...")

        tts = gTTS(text, lang='en')
        tts.save(TTS_FILE)

        return TTS_FILE  # Return the path to the generated audio file
    except ImportError:
        logger.error("`gtts` library not found. Please install the GTTS library using: `pip install gtts`.")
    except Exception as e:
        logger.error(f"TTS failed - {e}")
        logger.info("Error occurred during TTS!")

    return None
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def edge_tts(text: str):
    try:
        import re
        logger.debug("Trying to convert Text to Speech using edge-tts...")

        voice = EDGE_VOICE_NAME
        clean_text = re.sub(r'[\\"]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        command = f'edge-tts --voice "{voice}" --text "{clean_text}" --write-media {TTS_FILE}'
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        await process.communicate()
        await process.wait()

        return TTS_FILE  # Return the path to the generated audio file
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        logger.info("Error occurred during TTS!")

    return None
# ----------------------------------------------------------------------------