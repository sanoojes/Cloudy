import shutil
import subprocess
import asyncio
import io

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
            quit()
        except ModuleNotFoundError:
            message = "`pip install pygame` required when `use_ffmpeg=False`"
            raise ValueError(message)
        finally:
            pygame.mixer.quit()
            pygame.quit()

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