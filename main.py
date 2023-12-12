# -*- coding: utf-8 -*-
"""
AI Conversational System 
Codename: Cloudy

This script implements an AI conversational system utilizing OpenAI's GPT-3.5 language model.
The system interacts with users through speech recognition, AI responses, and facial recognition.

Author: Sachu-Settan
"""
CONFIG_TYPE = 'config' # 'config' for config.json and 'env' for dotenv file
# ----------------------------------------------------------------------------
import contextlib
import math
import subprocess

with contextlib.redirect_stdout(None):
    import re
    import os
    import cv2
    import time
    import json
    import shutil
    import pygame
    import asyncio
    import logging
    import aiohttp
    import schedule
    import numpy as np
    from gtts import gTTS
    import face_recognition
    from openai import OpenAI
    from dotenv import load_dotenv
    import speech_recognition as sr
    from collections import Counter, deque
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
LOGGING_LEVEL = logging.INFO
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Load environment variables from .env file.If present
load_dotenv()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Load configurations from JSON file
def load_config(CONFIG_FILE):
    """
    Loads configurations from a JSON file.

    Parameters:
    - CONFIG_FILE (str): Path to the configuration JSON file.

    Returns:
    - dict: Loaded configurations.
    """
    try:
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError as e:
        logger.error("Config file not found at specified path: %s", CONFIG_FILE)
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}") from e
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Set up logging
def setup_logging(log_file):
    """
    Sets up logging configuration.

    Parameters:
    - log_file (str): Path to the log file.
    """
    logging.basicConfig(level=LOGGING_LEVEL, format="%(levelname)s - %(message)s")
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file, mode="a")  # Append mode
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Schedule log clearing task every 10 minutes
    schedule.every(10).minutes.do(clear_log_file, log_file)
    return logger
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def clear_log_file(log_file):
    """
    Clears the log file at regular intervals.

    Parameters:
    log_file (str): Path to the log file.
    """
    try:
        with open(log_file, "w"):
            pass  # Clear the log file by opening it in write mode
        logger.debug(
            f"Cleared log file '{log_file}' at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        logger.exception(f"Error occurred while clearing log file: {e}")
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function to save data to a JSON file
def save_json(data, filename):
    """
    Saves data to a JSON file.

    Parameters:
    - data (dict): Data to be saved.
    - filename (str): Path to the JSON file.
    """
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function to load data from a JSON file
def load_json(filename):
    """
    Loads data from a JSON file.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - dict: Loaded data.
    """
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function to get AI response from OpenAI
def get_response_from_openai(user_input, messages):
    """
    Gets AI response from OpenAI.

    Parameters:
    - user_input (str): User's input/query.
    - messages (list): List of messages in the conversation.

    Returns:
    - str: AI response.
    """
    cache_key = user_input.lower()
    try:
        chat_reply = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )
        ai_response = chat_reply.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_response})
        return ai_response
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        quit()
    except Exception as e:
        logger.debug(f"Error: Unexpected error occurred:{e}")
        return "Error: Unexpected error occurred check your openAI API key."
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def convert_text_to_audio(text):
    """
    Converts the provided text into an audio file using the Edge-TTS library.

    Parameters:
    text (str): Text to convert into audio.

    Returns:
    Tuple[str, bytes] or None: Tuple containing path to the generated audio file and command output,
                               or None if there's an error.
    """
    try:
        voice = EDGE_VOICE_NAME
        clean_text = re.sub(r"[^a-zA-Z0-9\s.]", "", text)
        path = TTS_FILE
        command = (
            f'edge-tts --voice "{voice}" --text "{clean_text}" --write-media {path}'
        )

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            # Log the error if the command fails
            print(f"Error occurred: {stderr.decode('utf-8')}")
            return None

        return path  # Return the path to the generated audio file and command output

    except Exception as e:
        print("TTS failed:", e)
        return None


async def play_audio(audio_file):
    """
    Plays the audio file using the Pygame library asynchronously.

    Parameters:
    audio_file (str): Path to the audio file.
    """
    try:
        logger.debug("Playing Audio...")
        pygame.init()
        pygame.mixer.init()

        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

    except (pygame.error, IOError) as e:
        logger.error("Error during audio playback:", e)
        # Handle errors related to audio playback here.
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        quit()
    except Exception as e:
        logger.error("Unexpected error during audio playback:", e)
        # Handle any other unexpected errors here.
    finally:
        pygame.mixer.quit()
        pygame.quit()
        # Remove the temporary audio file
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)


async def play_text_as_audio(text):
    """
    Converts the provided text into audio and plays it asynchronously.

    Parameters:
    - text (str): The text to be converted into audio and played.

    Returns:
    - None
    """
    try:
        audio_file = await convert_text_to_audio(text)
        if audio_file:
            await play_audio(audio_file)
    except Exception as e:
        logger.error("Unexpected error during text to audio:", e)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function to play audio with GTTS library
#
# def play_text_as_audio(text):
#     try:
#         pygame.init()
#         pygame.mixer.init()

#         try:
#             logger.debug("Trying to convert Text to Speech...")
#             tts = gTTS(text=text, lang="en")
#         except TimeoutError:
#             logger.debug("TTS failed",TimeoutError)
#             logger.info("Check Your Internet. It might be slow !")
#         except Exception as e:
#             logger.debug("TTS failed",e)
#             logger.info("Error Occurred During TTS !")

#         # Create a temporary file to store the audio
#         temp_audio_file = tempfile.NamedTemporaryFile(delete=False)
#         tts.write_to_fp(temp_audio_file)
#         temp_audio_file.close()

#         pygame.mixer.music.load(temp_audio_file.name)
#         pygame.mixer.music.play()

#         # Wait for the audio to finish playing
#         while pygame.mixer.music.get_busy():
#             time.sleep(0.1)

#     except (pygame.error, IOError) as e:
#         logger.info("Error during audio playback:", e)
#         # Handle errors related to audio playback here.

#     except KeyboardInterrupt:
#         logger.info("Keyboard interrupt detected")
#         quit()

#     except Exception as e:
#         logger.info("Unexpected error during audio playback:", e)
#         # Handle any other unexpected errors here.

#     finally:
#         pygame.mixer.quit()
#         pygame.quit()
#         # Remove the temporary audio file
#         os.unlink(temp_audio_file.name)
#
# ----------------------------------------------------------------------------

# #----------------------------------------------------------------------------
# # Functions to play audio with edge-tts library for realistic voice
# def convert_text_to_audio(text):
#     """
#     Converts the provided text into an audio file using the Edge-TTS library.

#     Parameters:
#     text (str): Text to convert into audio.
#     Returns:
#     str: Path to the generated audio file."""
#     voice = EDGE_VOICE_NAME
#     clean_text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
#     command = (
#         f'edge-tts --voice "{voice}" --text "{clean_text}" --write-media "./src/data.mp3"'
#     )

#     try:
#         logger.debug("Trying to convert Text to Speech...")
#         subprocess.call(command, shell=False)
#     except Exception as e:
#         logger.debug("TTS failed", e)
#         logger.info("Error Occurred During TTS !")
#         # Handle TTS failure if needed
#         return None

#     return "./src/data.mp3"  # Return the path to the generated audio file

# def play_audio(audio_file):
#     '''
#     Plays the audio file using the Pygame library.

#     Parameters:
#     audio_file (str): Path to the audio file.
#     '''
#     try:
#         logging.debug("Playing Audio...")
#         pygame.init()
#         pygame.mixer.init()

#         pygame.mixer.music.load(audio_file)
#         pygame.mixer.music.play()

#         while pygame.mixer.music.get_busy():
#             time.sleep(0.1)

#     except (pygame.error, IOError) as e:
#         logger.error("Error during audio playback:", e)
#         # Handle errors related to audio playback here.
#     except KeyboardInterrupt:
#         logger.info("Keyboard interrupt detected")
#         quit()
#     except Exception as e:
#         logger.error("Unexpected error during audio playback:", e)
#         # Handle any other unexpected errors here.
#     finally:
#         pygame.mixer.quit()
#         pygame.quit()
#         # Remove the temporary audio file
#         if audio_file and os.path.exists(audio_file):
#             os.remove(audio_file)

# def play_text_as_audio(text):
#     '''
#     Converts the provided text into audio and plays it.

#     Parameters:
#     - text (str): The text to be converted into audio and played.

#     Returns:
#     - None
#     '''
#     try:
#         audio_file = convert_text_to_audio(text)
#         if audio_file:
#             play_audio(audio_file)
#     except Exception as e:
#         logger.error("Unexpected error during text to audio:", e)
# #----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def get_google_desc(query):
    """
    Fetches search results from Google for the given query.

    Parameters:
    query (str): Query to search on Google.
    Returns:
    str: Description or snippet obtained from Google search results.
    """
    base_url = "https://www.googleapis.com/customsearch/v1"

    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                response.raise_for_status()  # Raises an exception for 4xx/5xx status codes

                data = await response.json()

                # Extracting search results
                if "items" in data:
                    search_results = data["items"]
                    data2 = ""
                    for i in search_results:
                        data2 += i["snippet"]
                    logger.debug("Google result: " + data2)
                    return data2
                else:
                    logger.debug("No search results found.")

    except aiohttp.ClientError as e:
        logger.exception(f"Client error occurred: {e}")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def detect_question(input_string):
    """
    Detects if a question is present in the input string.

    Parameters:
    - input_string (str): Input string to be analyzed.

    Returns:
    - bool: True if a question is detected, False otherwise.
    """
    question_keywords = ["who", "what", "when", "where", "why", "how", "tell"]
    for keyword in question_keywords:
        if keyword in input_string.lower():
            return True
    return False
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Optimized speech recognition function
def recognize_speech(recognizer, source):
    """
    Recognizes speech using the provided speech recognizer.

    Parameters:
    - recognizer (Recognizer): Speech recognizer object.
    - source (Microphone): Microphone source for input.

    Returns:
    - str: Recognized speech text.
    """

    attempts = 0
    recognizer.adjust_for_ambient_noise(source)
    
    recognition_methods = {
        "google": recognizer.recognize_google,
        "vosk": lambda audio: json.loads(recognizer.recognize_vosk(audio))['text'],
        "sphinx": recognizer.recognize_sphinx,
        "whisper": recognizer.recognize_whisper  # Assuming this method exists
    }
    logger.info("Listening...")
    while True:
        attempts += 1
        logger.debug("Attempt %d to recognize speech...", attempts)

        try:
            audio = recognizer.listen(source)

            if audio:
                if recognizer.energy_threshold:
                    # Adjust energy threshold based on audio characteristics
                    rms_value = math.sqrt(sum([sample**2 for sample in audio.frame_data]) / len(audio.frame_data))
                    recognizer.energy_threshold = rms_value + 100

                # Adjust other recognition parameters if needed

                recognition_result = None

                recognizer_func = recognition_methods.get(RECOGNIZER_MODEL)
                if recognizer_func:
                    recognition_result = recognizer_func(audio)
                else:
                    logging.error("Recognizer model not set or not recognized.")
                    raise ValueError("Recognizer model not set or not recognized.")

                if recognition_result:
                    return recognition_result
                else:
                    logger.info("Recognition result is empty.")
                    continue
            else:
                logger.info("No audio input received.")

        except sr.WaitTimeoutError:
            logger.info("Listening timed out. Please speak louder or try again.")
        except sr.UnknownValueError:
            logger.info("Could not understand audio.")
        except sr.RequestError as e:
            logger.error("Error requesting from Speech Recognition service: %s", e)
            time.sleep(1)  # Wait for a while before retrying
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected")
            quit()
        except Exception as e:
            logger.exception("Error occurred: %s", e)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Load messages from a JSON file
def load_messages(filename):
    """
    Loads conversation messages from a JSON file.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - list: Loaded conversation messages.
    """
    try:
        logger.debug("Loading Messages File...")
        with open(filename, "r") as file:
            logger.debug("Success Loaded Messages File")
            return json.load(file)
    except FileNotFoundError:
        logger.debug("File specified not found.")
        return []
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def get_offline_response_file(username):
    """
    Generates the file path for storing offline AI responses based on the username.

    Parameters:
    - username (str): The username used to create the file path.

    Returns:
    - str: The file path for storing offline AI responses based on the username.
    """
    if not os.path.exists(OFFLINE_DATA_FOLDER):
        os.makedirs(OFFLINE_DATA_FOLDER)
    filename = username + "_offline_data.json"
    file = os.path.join(OFFLINE_DATA_FOLDER, filename)
    return file
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def get_messages_file(username):
    """
    Gets the file path for storing conversation messages based on the username.

    Parameters:
    - username (str): The username used to generate the file path.

    Returns:
    - str: The file path for storing conversation messages based on the username.
    """
    if not os.path.exists(MESSAGES_DATA_FOLDER):
        os.makedirs(MESSAGES_DATA_FOLDER)
    filename = username + "_messages_data.json"
    file = os.path.join(MESSAGES_DATA_FOLDER, filename)
    return file
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def load_default_offline_ai_response(offline_user_file):
    """
    Loads offline AI responses from a JSON file.

    Parameters:
    - offline_user_file (str): Path to the offline AI responses JSON file.

    Returns:
    - dict: Loaded offline AI responses.
    """
    global offline_ai_response
    try:
        offline_ai_response = load_json(offline_user_file)
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", e)
    finally:
        return offline_ai_response
# ----------------------------------------------------------------------------

# Load Default Offline Responce
default_offline_responce = load_default_offline_ai_response(
    "training/offline/default_offline.json"
)

# ----------------------------------------------------------------------------
def load_offline_ai_response(offline_user_file):
    """
    Loads offline AI responses from a JSON file.

    Parameters:
    - offline_user_file (str): Path to the offline AI responses JSON file.

    Returns:
    - dict: Loaded offline AI responses.
    """
    global offline_ai_response
    try:
        offline_ai_response = load_json(offline_user_file)
        offline_ai_response.update(default_offline_responce)
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", e)
    finally:
        return offline_ai_response
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def handle_conversation(recognizer, username):
    offline_user_file = get_offline_response_file(username)
    offline_ai_response = load_offline_ai_response(offline_user_file)
    messages = []  # Initialize messages or use an appropriate data structure

    try:
        while True:
            schedule.run_pending()
            with sr.Microphone() as source:
                continue_conversation = await process_user_input(
                    recognizer, source, offline_ai_response, messages
                )
                if not continue_conversation:
                    break
    except KeyboardInterrupt:
        logger.info("Conversation terminated by user.")
        quit()
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", e)
    finally:
        logger.info("End of conversation.")
        quit()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def process_user_input(recognizer, source, offline_ai_response, messages):
    """
    Processes the user's input, recognizes speech, and generates AI responses.

    Parameters:
    - recognizer (Recognizer): Speech recognizer object.
    - source (Microphone): Microphone source for input.
    - offline_ai_response (dict): Dictionary storing offline AI responses.
    - messages (list): List of conversation messages.

    Returns:
    - bool: Indicates whether the conversation should continue or not.
    """
    try:
        speech = recognize_speech(recognizer, source)
        if speech == None:
            speech = " "
        speech = speech.lower()
        if speech:
            if "rowdy" in speech.lower():
                speech = speech.replace("rowdy", "cloudy")
            logger.info("You said: %s", speech)
            if "cloudy" in speech.lower():
                if "cloudy change user" in speech.lower():
                    logging.info("Trying To change user..")
                    asyncio.run(play_text_as_audio("Trying To change user"))
                    detected_person = await load_user()
                    offline_ai_response.clear()
                    global offline_user_file
                    offline_user_file = get_offline_response_file(detected_person)
                    offline_ai_response = load_offline_ai_response(offline_user_file)
                    global username
                    username = detected_person
                    messages.clear()
                    messages = load_messages(get_messages_file(detected_person))
                    global ai_instructions
                    ai_instructions = generate_ai_instructions(detected_person)
                    play_greeting_audio(detected_person)
                else:
                    messages.append({"role": "system", "content": ai_instructions})
                    if detect_question(speech):
                        s = speech.replace("cloudy", " ")
                        s = s.replace("rowdy", " ")
                        a = await get_google_desc(s)
                        mess = f"\nuser query: {speech}\n\nsearch results:{a}\n"
                        messages.append({"role": "user", "content": mess})
                    else:
                        messages.append({"role": "user", "content": speech})
                    save_json(messages, get_messages_file(username))
                    response_from_openai = offline_ai_response.get(speech.lower(), None)

                    if not response_from_openai:
                        response_from_openai = get_response_from_openai(
                            speech, messages
                        )
                        offline_ai_response[speech.lower()] = response_from_openai
                        save_json(
                            offline_ai_response, get_offline_response_file(username)
                        )

                    logger.info("cloudy: %s", response_from_openai)
                    await play_text_as_audio(response_from_openai)

            elif "turn off" == speech.lower():
                logger.info("Exiting...")
                return False

        else:
            logger.info("No speech detected. Please speak louder or try again.")

        return True
    except KeyboardInterrupt:
        logger.info("Conversation terminated by user.")
        quit()
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", e)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def load_user():
    """
    Detects the user's face and returns the detected name.

    Returns:
    - str: Detected username.
    """
    logger.info("Detecting Face....")
    detected_name = await run_face_recognition()
    logging.info("Detected User: " + detected_name)
    return detected_name
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def save_facial_data(known_face_encodings, known_face_names):
    if not os.path.exists(facial_data_folder):
        os.makedirs(facial_data_folder)
    # Convert NumPy arrays to lists before serialization
    known_face_encodings_serializable = [
        encoding.tolist() for encoding in known_face_encodings
    ]

    data = {"encodings": known_face_encodings_serializable, "names": known_face_names}

    with open(os.path.join(facial_data_folder, FACIAL_DATA_FILE), "w") as file:
        json.dump(data, file)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def load_facial_data():
    data_file = os.path.join(facial_data_folder, FACIAL_DATA_FILE)
    if os.path.exists(data_file):
        with open(data_file, "r") as file:
            data = json.load(file)
            known_face_encodings = [
                np.array(encoding) for encoding in data["encodings"]
            ]
            known_face_names = data["names"]
            return known_face_encodings, known_face_names
    return [], []
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def recognize_faces(video_capture, known_face_encodings, known_face_names):
    try:
        detections = deque(
            maxlen=5
        )  # Using deque for storing detections with a maximum length
        detected_name = None

        register_face = None  # Initialize register_face outside the loop

        while True:
            schedule.run_pending()
            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"

                if True in matches:
                    matched_index = matches.index(True)
                    name = known_face_names[matched_index]
                else:
                    await play_text_as_audio(
                            "Hello there! Would you like to register? (Press 'y' to register)"
                        )
                    logging.info(
                        "Hello there! Would you like to register? (Press 'y' to register)"
                    )
                    register_face = input().lower()

                    if register_face == "y":
                        await play_text_as_audio("Please enter your name:")
                        logging.info("Please enter your name:")
                        new_name = input()
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(new_name)
                        name = new_name
                        play_text_as_audio("User registration success !")
                        logging.info("User registration successful ! Username: " + name)
                    else:
                        name = "Unknown"

                face_names.append(name)

            detections.append(face_names)

            # Check for consistent detections in the last 5 frames
            occurrences = Counter(name for sublist in detections for name in sublist)
            detected_name = next(
                (name for name, count in occurrences.items() if count >= 3), None
            )

            if detected_name:
                detections.clear()
                return detected_name

            if register_face == "y":
                save_facial_data(known_face_encodings, known_face_names)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        quit()
    except Exception as e:
        logger.info("Unexpected error during face recognition:", e)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Modify the run_face_recognition function to receive the detected name
async def run_face_recognition():
    try:
        video_capture = cv2.VideoCapture(0)
        known_face_encodings, known_face_names = load_facial_data()
        detected_name = await recognize_faces(
            video_capture, known_face_encodings, known_face_names
        )
        video_capture.release()
        if detected_name is None:
            detected_name = "User"
        return detected_name
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected during face recognition.")
        quit()
    except Exception as e:
        logger.info("Unexpected error during face recognition:", e)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function to play a greeting audio for the detected user's name
async def play_greeting_audio(detected_person):
    try:
        greeting_text = f"Hi {detected_person}! How can I assist you today?"
        logging.info("Greating User...")
        logging.info("cloudy: " + greeting_text)
        await play_text_as_audio(greeting_text)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        quit()
    except Exception as e:
        logger.info("Unexpected error during greeting audio:", e)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function to generate an AI message based on the detected user's name
def generate_ai_instructions(username):
    global ai_instructions
    ai_instructions = AI_INS
    ai_instructions = ai_instructions.format(user=username)
    return ai_instructions
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def cleanup_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            # Perform asynchronous deletion of files within the folder
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

            # Asynchronously remove the empty directories
            await asyncio.to_thread(shutil.rmtree, folder_path)
            logging.debug(f"Folder '{folder_path}' deleted.")
        else:
            logging.debug(f"Folder '{folder_path}' does not exist.")
    except PermissionError as pe:
        logging.debug(
            f"Permission error occurred: {pe}. Skipping folder '{folder_path}'."
        )
    except KeyboardInterrupt:
        logger.info("Conversation terminated by user.")
        quit()
    except Exception as e:
        logging.error(f"An error occurred during cleanup of '{folder_path}': {e}")

async def cleanup_files(folders_to_cleanup):
    try:
        current_directory = os.getcwd()
        logging.debug(f"Current working directory: {current_directory}")

        # Asynchronously clean up each folder
        cleanup_tasks = [
            cleanup_folder(folder_path) for folder_path in folders_to_cleanup
        ]
        await asyncio.gather(*cleanup_tasks)
    except KeyboardInterrupt:
        logger.info("Conversation terminated by user.")
        quit()
    except Exception as e:
        logging.error(f"An error occurred during cleanup: {e}")
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class NoInternetConnection(Exception):
    pass


async def check_internet_connection():
    try:
        logging.info("Checking Your Internet Connection...")
        async with aiohttp.ClientSession() as session:
            async with session.get(CHECK_INTERNET_URL) as response:
                if response.status == 200:
                    logging.info("Internet is connected")
                else:
                    logging.info("Internet is not connected")
                    raise NoInternetConnection("Internet is not connected")
    except aiohttp.ClientError:
        logging.info("No internet connection available")
        raise NoInternetConnection("No internet connection available")
    except KeyboardInterrupt:
        logger.info("Conversation terminated by user.")
        quit()
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Load environment variables from config.json or .env
CONFIG_FILE = "config.json"
if CONFIG_TYPE == "config":
    print("Loading Configurations from config.json")
    config = load_config(CONFIG_FILE)
    LOG_FILE = config.get("LOG_FILE")
    TRAINING_FOLDER = config.get("TRAINING_FOLDER")
    OFFLINE_DATA_FILE = config.get("OFFLINE_DATA_FILE")
    FACIAL_DATA_FOLDER = config.get("FACIAL_DATA_FOLDER")
    FACIAL_DATA_FILE = config.get("FACIAL_DATA_FILE")
    OFFLINE_DATA_FOLDER = config.get("OFFLINE_DATA_FOLDER")
    MESSAGES_DATA_FOLDER = config.get("MESSAGES_DATA_FOLDER")
    EDGE_VOICE_NAME = config.get("EDGE_VOICE_NAME")
    GOOGLE_API_KEY = config.get("GOOGLE_API_KEY")
    GOOGLE_CX = config.get("GOOGLE_CX")
    RECOGNIZER_MODEL = config.get("RECOGNIZER_MODEL")
    CHECK_INTERNET_URL = config.get("CHECK_INTERNET_URL")
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    TTS_FILE = config.get("TTS_FILE")
elif CONFIG_TYPE == "env":
    print("Loading Configurations from .env file")
    LOG_FILE = os.getenv("LOG_FILE")
    TRAINING_FOLDER = os.getenv("TRAINING_FOLDER")
    OFFLINE_DATA_FILE = os.getenv("OFFLINE_DATA_FILE")
    FACIAL_DATA_FOLDER = os.getenv("FACIAL_DATA_FOLDER")
    FACIAL_DATA_FILE = os.getenv("FACIAL_DATA_FILE")
    OFFLINE_DATA_FOLDER = os.getenv("OFFLINE_DATA_FOLDER")
    MESSAGES_DATA_FOLDER = os.getenv("MESSAGES_DATA_FOLDER")
    EDGE_VOICE_NAME = os.getenv("EDGE_VOICE_NAME")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CX = os.getenv("GOOGLE_CX")
    RECOGNIZER_MODEL = os.getenv("RECOGNIZER_MODEL")
    CHECK_INTERNET_URL = os.getenv("CHECK_INTERNET_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TTS_FILE = os.getenv("TTS_FILE")
# ----------------------------------------------------------------------------
folders_to_cleanup = [MESSAGES_DATA_FOLDER]
# ----------------------------------------------------------------------------
offline_ai_response = {}
# ----------------------------------------------------------------------------
facial_data_folder = os.path.join(TRAINING_FOLDER, FACIAL_DATA_FOLDER)
# ----------------------------------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
AI_INS = "Your AI name is Cloudy, and the user's name is {user}.\
You were developed by Sachu and a team of students from st josephs public school and are designed to provide concise answers in 2,3 lines.\
You serve as an interactive educational robot. \
If the search results are not relevent:\
- Interact with user\
- Do not tell {user} about search results\
Only User Query is provided by the user\
When provided with up-to-date search results from a user query:\
- Dont tell user to verify it from reliable sources as we are providing up to date information\
- Ask to clarify their question if the query and search results are not relevent.\
- Check the relevance of the results to the query.\
- If the results make sense (are relevant), summarize them concisely.\
- If the results seem irrelevant or confusing in relation to the query, engage the user in conversation."
ai_instructions = AI_INS
# ----------------------------------------------------------------------------
# Set up logging
logger = setup_logging(LOG_FILE)
# ----------------------------------------------------------------------------
# Load messages
messages = load_messages(get_messages_file("default"))
# ----------------------------------------------------------------------------
# Load Default Offline messages
offline_user_file = get_offline_response_file("default")
username = "User"
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def main():
    """Main function to run the conversational AI system."""
    try:
        await check_internet_connection()
        await cleanup_files(folders_to_cleanup)
        recognizer = sr.Recognizer()
        detected_person = await load_user()
        global username
        username = detected_person
        global messages
        messages.clear()
        messages = load_messages(get_messages_file(detected_person))
        global ai_instructions
        ai_instructions = generate_ai_instructions(detected_person)
        global offline_user_file
        offline_user_file = get_offline_response_file(detected_person)
        await play_greeting_audio(detected_person)
        while True:
            schedule.run_pending()
            continue_conversation = await handle_conversation(recognizer, detected_person)
            if not continue_conversation:
                break
    except NoInternetConnection:
        exit()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
# ----------------------------------------------------------------------------