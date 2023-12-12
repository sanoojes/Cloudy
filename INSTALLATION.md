# **Installation Guide for Cloudy**

## **Python Installation (Use 3.9 or 3.9+)**

### **Windows**

1. Download the latest Python installer for Windows from [python.org](https://www.python.org/downloads/windows/).
2. Run the installer and select the option to add Python to PATH.
3. Follow the installation instructions, and Python will be installed on your system.

### **Linux**

1. Python is generally pre-installed on most Linux distributions. Use the package manager specific to your distribution to install Python if it's not already installed. For example, on Ubuntu:

    ```bash
    sudo apt update
    sudo apt install python3.9
    ```

## **Package Installation**

Open a terminal or command prompt and use `pip` to install the required packages.

#### **For Windows**

```bash
# pip install cmake
# pip install dlib 
# If you got error in dlib use precompiled dlib wheels
# https://github.com/Sachu-Settan/dlib 
# pip install opencv-python-headless
# pip install numpy
# pip install pygame
# pip install aiohttp
# pip install gTTS
# pip install face-recognition
# pip install openai
# pip install pyaudio
# pip install python-dotenv
# pip install SpeechRecognition
```

#### **For Linux**

```bash
# sudo apt-get install python3-opencv
```

```bash
# pip install cmake
# pip install dlib
# pip install opencv-python-headless
# pip install numpy
# pip install pygame
# pip install aiohttp
# pip install gTTS
# pip install pyaudio
# pip install face-recognition
# pip install openai
# pip install python-dotenv
# pip install SpeechRecognition
```

## **Additional Steps**

### **Setting up Secrets and Configuration** 
### **NOTE:** Please Select config or dotenv before setting up your default variables in [main.py](main.py#11)

#### Setting up [Config.json](config.json)

```bash
{
    "OPENAI_API_KEY":"Your OpenAI API key",
    "LOG_FILE" :"db/logs.log",
    "GOOGLE_CX":"Your Google CX Key For Programmable Custom Search Engine",
    "GOOGLE_API_KEY":"Your Google API Key",
    "TRAINING_FOLDER":"training",
    "FACIAL_DATA_FOLDER":"face_data",
    "FACIAL_DATA_FILE":"facial_data.json",
    "OFFLINE_DATA_FOLDER":"training//offline",
    "MESSAGES_DATA_FOLDER":"db/messages",
    "EDGE_VOICE_NAME":"en-US-JennyNeural",
    "RECOGNIZER_MODEL":"google",
    "CHECK_INTERNET_URL":"https://1.1.1.1",
    "TTS_FILE": "./src/data.mp3"
} 
```

#### Setting up [.env](.env)

```bash
OPENAI_API_KEY = Your OpenAI API Key
LOG_FILE = db/logs.log
GOOGLE_CX =  Your Google CX Key
GOOGLE_API_KEY = Your Google API Key
TRAINING_FOLDER = training    
FACIAL_DATA_FOLDER = face_data
FACIAL_DATA_FILE = facial_data.json
OFFLINE_DATA_FOLDER = training//offline
MESSAGES_DATA_FOLDER = db/messages
EDGE_VOICE_NAME = en-US-JennyNeural
RECOGNIZER_MODEL = google
CHECK_INTERNET_URL = https://1.1.1.1
TTS_FILE = ./src/data.mp3
```

### **Setting Up Environment**

#### **For Windows**

[`CMake`](https://cmake.org/):` Download the CMake installer for Windows from cmake.org and follow the installation instructions.`

#### **For Linux**

CMake: Install CMake using your package manager. For example, on Ubuntu:

```bash
sudo apt update && sudo apt upgrade
sudo apt install cmake
```

### **Optional Libraries**

- [**os**](https://docs.python.org/3/library/os.html)
- [**math**](https://docs.python.org/3/library/math.html)
- [**time**](https://docs.python.org/3/library/time.html)
- [**json**](https://docs.python.org/3/library/json.html)
- [**shutil**](https://docs.python.org/3/library/shutil.html)
- [**asyncio**](https://docs.python.org/3/library/asyncio.html)
- [**logging**](https://docs.python.org/3/library/logging.html)
- [**schedule**](https://schedule.readthedocs.io/en/stable/)
- [**subprocess**](https://docs.python.org/3/library/subprocess.html)
- [**collections**](https://docs.python.org/3/library/collections.html)

**These are Python standard libraries and are usually included by default with Python. No separate installation is required.**