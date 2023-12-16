import os
import cv2
import sys
import pickle
import face_recognition
from collections import Counter
from brain.play import play_text_as_audio
from main import logger, facial_data_folder, FACIAL_DATA_FILE

# Constants
MAX_RETRIES = 5
MAX_NAME_LENGTH = 15

# Function to save facial data to a file
def save_facial_data(known_face_encodings, known_face_names):
    data = {"encodings": known_face_encodings, "names": known_face_names}
    file_path = os.path.join(facial_data_folder, FACIAL_DATA_FILE)

    try:
        if not os.path.exists(facial_data_folder):
            os.makedirs(facial_data_folder)

        with open(file_path, "wb") as file:
            pickle.dump(data, file)
    except Exception as e:
        logger.error(f"Failed to save facial data: {e}")

# Function to load facial data from a file
def load_facial_data():
    data_file = os.path.join(facial_data_folder, FACIAL_DATA_FILE)
    known_face_encodings, known_face_names = [], []

    try:
        if os.path.exists(data_file):
            with open(data_file, "rb") as file:
                data = pickle.load(file)
                known_face_encodings = data.get("encodings", [])
                known_face_names = data.get("names", [])
    except Exception as e:
        logger.error(f"Failed to load facial data: {e}")

    return known_face_encodings, known_face_names

# Function to recognize faces
async def recognize_faces(video_capture, known_face_encodings, known_face_names):
    try:
        detections = []

        while True:
            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    matched_index = matches.index(True)
                    name = known_face_names[matched_index]
                else:
                    name = await register_new_user(face_encoding, known_face_encodings, known_face_names)
                    logger.info("Detecting Face....")

                face_names.append(name)

            detections.append(face_names)

            occurrences = Counter(name for sublist in detections for name in sublist)
            detected_name = next((name for name, count in occurrences.items() if count >= 3), None)

            if detected_name:
                detections.clear()
                return detected_name

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error during face recognition:", exc_info=True)

# Function to register a new user
async def register_new_user(face_encoding, known_face_encodings, known_face_names):
    retries = MAX_RETRIES

    while retries > 0:
        try:
            logger.info("Hello there! Would you like to register? (Press 'y' to register)")
            await play_text_as_audio("Hello there! Would you like to register? (Press 'y' to register)")
            register_face = input().lower()

            if register_face in ["y", "yes"]:
                name = await register_name()
                if name and is_valid_name(name):
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    save_facial_data(known_face_encodings=known_face_encodings,known_face_names=known_face_names)
                    logger.info(f"User registration successful! Username: {name}")
                    await play_text_as_audio(f"User registration successful! Username: {name}")
                    return name
                else:
                    logger.info("Invalid name format. Please enter a single-word name (up to 15 characters).")
                    await play_text_as_audio("Invalid name format. Please enter a single-word name (up to 15 characters).")
                    retries -= 1
            elif register_face in ["n", "no"]:
                logger.info("You declined to register.")
                await play_text_as_audio("You declined to register.")
                break
            else:
                logger.info("Invalid input. Please enter 'y' or 'n' to register or decline.")
                await play_text_as_audio("Invalid input. Please enter 'y' or 'n' to register or decline.")
                retries -= 1

        except Exception as e:
            logger.error("Error in registering name:", exc_info=True)
            await play_text_as_audio("An error occurred while registering. Please try again.")
            retries -= 1

    logger.warning("Maximum retries reached. Proceeding as a default unknown user.")
    await play_text_as_audio("Maximum retries reached. Proceeding as a default unknown user.")
    return "User"

# Function to register a name for a new user
async def register_name():
    try:
        logger.info("Please enter your name:")
        await play_text_as_audio("Please enter your name:")
        new_name = input().lower()
        return new_name if is_valid_name(new_name) else None
    except Exception as e:
        logger.error("Error in registering name:", exc_info=True)
        await play_text_as_audio("An error occurred while registering. Please try again.")
        return None

# Function to validate the name format
def is_valid_name(name):
    return name.replace(" ", "").isalpha() and len(name.split()) == 1 and len(name) <= MAX_NAME_LENGTH

# Main function to run face recognition
async def run_face_recognition():
    try:
        logger.info("Detecting Face....")
        video_capture = cv2.VideoCapture(0)
        known_face_encodings, known_face_names = load_facial_data()
        detected_name = await recognize_faces(video_capture, known_face_encodings, known_face_names)
        video_capture.release()
        return detected_name if detected_name else "User"
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected during face recognition.")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error during face recognition:", exc_info=True)
