import os
import cv2
import pickle
import schedule
import face_recognition
from collections import Counter, deque
from main import logger, facial_data_folder, FACIAL_DATA_FILE, FACIAL_DATA_FOLDER
from .play import play_text_as_audio

# Function to save facial data to a file
def save_facial_data(known_face_encodings, known_face_names):
    if not os.path.exists(FACIAL_DATA_FOLDER):
        os.makedirs(FACIAL_DATA_FOLDER)

    data = {"encodings": known_face_encodings, "names": known_face_names}

    with open(os.path.join(facial_data_folder, FACIAL_DATA_FILE), "wb") as file:
        pickle.dump(data, file)

# Function to load facial data from a file
def load_facial_data():
    data_file = os.path.join(facial_data_folder, FACIAL_DATA_FILE)
    if os.path.exists(data_file):
        with open(data_file, "rb") as file:
            data = pickle.load(file)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
            return known_face_encodings, known_face_names
    return [], []

# Function to recognize faces in a video stream
async def recognize_faces(video_capture, known_face_encodings, known_face_names):
    try:
        detections = deque(maxlen=5)  # Store detections with a maximum length
        detected_name = None
        register_face = None  # Initialize register_face outside the loop

        while True:
            schedule.run_pending()
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
                    await play_text_as_audio("Hello there! Would you like to register? (Press 'y' to register)")
                    logger.info("Hello there! Would you like to register? (Press 'y' to register)")
                    register_face = input().lower()

                    if register_face == "y":
                        await play_text_as_audio("Please enter your name:")
                        logger.info("Please enter your name:")
                        new_name = input()
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(new_name)
                        name = new_name
                        await play_text_as_audio("User registration success!")
                        logger.info("User registration successful! Username: " + name)
                    else:
                        name = "Unknown"

                face_names.append(name)

            detections.append(face_names)

            # Check for consistent detections in the last 5 frames
            occurrences = Counter(name for sublist in detections for name in sublist)
            detected_name = next((name for name, count in occurrences.items() if count >= 3), None)

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

# Main function to run facial recognition
async def run_face_recognition():
    try:
        video_capture = cv2.VideoCapture(0)
        known_face_encodings, known_face_names = load_facial_data()
        detected_name = await recognize_faces(video_capture, known_face_encodings, known_face_names)
        video_capture.release()
        if detected_name is None:
            detected_name = "User"
        return detected_name
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected during face recognition.")
        quit()
    except Exception as e:
        logger.info("Unexpected error during face recognition:", e)
