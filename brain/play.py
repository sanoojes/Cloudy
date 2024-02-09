"""
AI Conversational System 
Codename: Cloudy

This script implements an AI conversational system utilizing OpenAI's GPT-3.5 language model.
The system interacts with users through speech recognition, AI responses, and facial recognition.

Author: Sachu-Settan i.e, Sanooj
Lisensed under GNU GPLv3 License.
"""

import shutil
import subprocess
import asyncio
import io
import os
import sys
from .tts import *
from main import logger

# ----------------------------------------------------------------------------
async def is_installed(lib_name: str) -> bool:
    return shutil.which(lib_name) is not None
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def play(audio_file: str, use_pygame: bool = False, use_ffmpeg: bool = True) -> None:
    async def play_sounddevice(audio_file):
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            message = "`pip install sounddevice soundfile` required when `use_ffmpeg=False`"
            raise ImportError(message)

        data, _ = await asyncio.to_thread(sf.read, audio_file)
        sd.play(data)
        sd.wait()
    async def pygame_play(audio_file):
        try:
            import pygame
            pygame.init()
            pygame.mixer.init()

            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

        except (pygame.error, IOError) as e:
            print("Error during audio playback:", e)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
            sys.exit(1)
        except ModuleNotFoundError:
            message = "`pip install pygame` required when `use_ffmpeg=False`"
            raise ValueError(message)
        finally:
            pygame.mixer.sys.exit(1)
            pygame.sys.exit(1)

    if use_pygame:
        await pygame_play(audio_file)

    elif use_ffmpeg:
    # Check if ffplay is installed
        if not await is_installed("ffplay"):
            message = (
            "ffplay from ffmpeg not found, necessary to play audio. "
            "On macOS, you can install it with 'brew install ffmpeg'. "
            "On Linux and Windows, you can install it from https://ffmpeg.org/"
        )
            raise ValueError(message)
    # Construct the command to play the audio file using ffplay
        command = f"ffplay -autoexit -nodisp {audio_file} -hide_banner -loglevel panic"
    # Execute the command and wait for it to complete
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).communicate()
    else:
        await play_sounddevice(audio_file)
# ----------------------------------------------------------------------------
        
# ----------------------------------------------------------------------------
async def play_bytes(audio: bytes, notebook: bool = False, use_ffmpeg: bool = True) -> None:
    if notebook:
        from IPython.display import Audio, display
        display(Audio(audio, rate=44100, autoplay=True))
    elif use_ffmpeg:
        if not await is_installed("ffplay"):
            message = (
                "ffplay from ffmpeg not found, necessary to play audio. "
                "On macOS, you can install it with 'brew install ffmpeg'. "
                "On Linux and Windows, you can install it from https://ffmpeg.org/"
            )
            raise ValueError(message)
        args = ["ffplay", "-hide_banner", "-loglevel", "error", "-autoexit", "-", "-nodisp"]
        proc = await asyncio.create_subprocess_exec(*args, stdin=subprocess.DEVNULL)
        await proc.communicate(input=audio)
    else:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ModuleNotFoundError:
            message = "`pip install sounddevice soundfile` required when `use_ffmpeg=False`"
            raise ValueError(message)

        data, _ = await asyncio.to_thread(sf.read, io.BytesIO(audio))
        sd.play(data)
        sd.wait()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def play_text_as_audio(text):
    """
    Converts the provided text into audio and plays it asynchronously.

    Parameters:
    - text (str): The text to be converted into audio and played.

    Returns:
    - None
    """
    try:
        audio_file = await edge_tts(text)  # Placeholder for TTS function
        # audio_file = await gtts(text)  # For GTTS, if needed
        if audio_file:    
            await play(audio_file)
    except Exception as e:
        logger.error("Unexpected error during text to audio:", e)

    if audio_file and os.path.exists(audio_file):
        os.remove(audio_file)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def play_greeting_audio(detected_person):
    try:
        greeting_text = f"Hi {detected_person}! How can I assist you today?"
        logger.info("Greeting User...")
        logger.info("cloudy: " + greeting_text)
        await play_text_as_audio(greeting_text)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error during greeting audio:", e)
# ----------------------------------------------------------------------------