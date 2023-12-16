import importlib
import sys
from io import StringIO

required_modules = [
    'os', 'random', 'datetime', 'schedule', 'openai', 'speech_recognition', 
    'dotenv', 'aiohttp', 'cv2', 'asyncio', 'gtts', 'edge_tts', 
    'cmake', 'face_recognition', 'pyaudio','pygame'
]

def check_modules(modules=required_modules):
    missing_modules = []
    
    # Redirect stdout to a null device
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    # Reset stdout back to its original value
    sys.stdout = original_stdout
    
    if missing_modules:
        print(f"Missing modules: {', '.join(missing_modules)}\nQuiting...")
        sys.exit(1)
    else:
        return "All required modules are available!"

