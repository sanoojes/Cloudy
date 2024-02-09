"""
AI Conversational System 
Codename: Cloudy

This script implements an AI conversational system utilizing OpenAI's GPT-3.5 language model.
The system interacts with users through speech recognition, AI responses, and facial recognition.

Author: Sachu-Settan i.e, Sanooj
Lisensed under GNU GPLv3 License.
"""

import os
import shutil
import asyncio
from main import logger

async def cleanup_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

            await asyncio.to_thread(shutil.rmtree, folder_path)
            logger.debug(f"Folder '{folder_path}' deleted.")
        else:
            logger.debug(f"Folder '{folder_path}' does not exist.")
    except PermissionError as pe:
        logger.debug(f"Permission error occurred: {pe}. Skipping folder '{folder_path}'.")
    except Exception as e:
        logger.error(f"An error occurred during cleanup of '{folder_path}': {e}")

async def cleanup_files(folders_to_cleanup):
    try:
        current_directory = os.getcwd()
        logger.debug(f"Current working directory: {current_directory}")

        cleanup_tasks = [
            cleanup_folder(folder_path) for folder_path in folders_to_cleanup
        ]
        await asyncio.gather(*cleanup_tasks)
    except Exception as e:
        logger.error(f"An error occurred during cleanup: {e}")
# ----------------------------------------------------------------------------
