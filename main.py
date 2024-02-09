"""
AI Conversational System 
Codename: Cloudy

This script implements an AI conversational system utilizing OpenAI's GPT-3.5 language model.
The system interacts with users through speech recognition, AI responses, and facial recognition.

Author: Sachu-Settan i.e, Sanooj
Lisensed under GNU GPLv3 License.
"""

from brain.check import check_modules
print(check_modules())

CONFIG_TYPE = 'config' # 'config' for config.json and 'env' for dotenv file
# ----------------------------------------------------------------------------
import contextlib

with contextlib.redirect_stdout(None):
    import os 
    import time
    import json
    import math 
    import asyncio
    import logging
    import schedule
    import speech_recognition as sr
    from brain import *
    from openai import OpenAI
    from dotenv import load_dotenv
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
LOGGING_LEVEL = logging.INFO
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Load environment variables from .env file.If present
load_dotenv()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Set up logging
def setup_logging(log_file):
    """
    Sets up logging configuration.

    Parameters:
    - log_file (str): Path to the log file.
    """
    logging.basicConfig(level=LOGGING_LEVEL, format="%(message)s")
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
def clear_log_file(log_file):
    """
    Clears the log file at regular intervals.

    Parameters:
    log_file (str): Path to the log file.
    """
    try:
        with open(log_file, "w"):
            pass  # Clear the log file by opening it in write mode
        print(
            f"Cleared log file '{log_file}' at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        print(f"Error occurred while clearing log file: {e}")
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
            sys.exit(1)
        except Exception as e:
            logger.exception("Error occurred: %s", e)
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
async def handle_conversation(recognizer, username):
    messages = []  # Initialize messages or use an appropriate data structure
    offline_ai_response = default_offline_responce
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
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", e)
    finally:
        logger.info("End of conversation.")
        sys.exit(1)
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
                    await play_text_as_audio("Trying To change user")
                    detected_person = await load_user()
                    offline_ai_response.clear()
                    global username
                    username = detected_person
                    messages.clear()
                    messages = load_messages(get_messages_file(detected_person))
                    global AI_INS
                    AI_INS = generate_ai_instructions(detected_person)
                    await play_greeting_audio(detected_person)
                else:
                    messages.append({"role": "system", "content": AI_INS})
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
                            messages
                        )
                        offline_ai_response[speech.lower()] = response_from_openai

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
        sys.exit(1)
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
    detected_name = await run_face_recognition()
    logging.info("Detected User: " + detected_name)
    return detected_name
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
    ELEVEN_LABS_VOICE =config.get("ELEVEN_LABS_VOICE")
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
    ELEVEN_LABS_VOICE = os.getenv("ELEVEN_LABS_VOICE")
# ----------------------------------------------------------------------------
folders_to_cleanup = [MESSAGES_DATA_FOLDER]
# ----------------------------------------------------------------------------
offline_ai_response = {}
# ----------------------------------------------------------------------------
facial_data_folder = os.path.join(TRAINING_FOLDER, FACIAL_DATA_FOLDER)
# ----------------------------------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
AI_INS =  """\
Hello! I'm Cloudy, your voice-based chat assistant, developed by Sachu and Vishnu. I communicate solely through voice interactions and aim to provide information without generating code snippets.

Guidelines for Interaction:
- Please ask questions or request information in a manner that doesn't prompt code generation.
- I focus on delivering voice-based responses without technical details or code snippets.
- If search results aren't relevant, I'll engage directly with voice-based information.
- When providing responses:
    - I'll emphasize voice-based details, avoiding technical explanations or code examples.
    - If needed, I'll ask for clarification to ensure accuracy in my voice-based answers.
    - I strive to offer concise and relevant information suitable for voice interaction.

I continuously update my knowledge to ensure the information I provide is current and accurate. Please ask queries in a way that aligns with voice-based responses!

User: My name is {user}.
"""
# ----------------------------------------------------------------------------
# Set up logging
logger = setup_logging(LOG_FILE)
# ----------------------------------------------------------------------------
# Load messages
messages = load_messages(get_messages_file("default"))
# ----------------------------------------------------------------------------
username = "User"
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
async def main():
    """Main function to run the conversational AI system."""
    try:
        connected = await check_internet_connection()
        if not connected:
        # Do something when there's no internet connection
            logger.info("Exiting application due to no internet connection.")
            exit(404)  # Use an appropriate exit code
        await cleanup_files(folders_to_cleanup)
        recognizer = sr.Recognizer()
        detected_person = await load_user()
        global username
        username = detected_person
        global messages
        messages.clear()
        messages = load_messages(get_messages_file(detected_person))
        global AI_INS
        AI_INS = generate_ai_instructions(detected_person)
        await play_greeting_audio(detected_person)
        while True:
            schedule.run_pending()
            continue_conversation = await handle_conversation(recognizer, detected_person)
            if not continue_conversation:
                break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Turning off the program")
        sys.exit(1)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Turning off the program")
        sys.exit(1)
# ----------------------------------------------------------------------------
