import aiohttp
from main import GOOGLE_API_KEY, GOOGLE_CX, logger, client, AI_INS
import re

# ----------------------------------------------------------------------------
# Function to generate AI instructions based on the detected user's name
def generate_ai_instructions(username):
    ai_instructions = AI_INS.format(user=username)
    return ai_instructions
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function to get AI response from OpenAI
def get_response_from_openai(messages):
    try:
        chat_reply = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )
        ai_response = chat_reply.choices[0].message.content

        # Clean the AI response
        clean_text = re.sub(r'\\[^\w\s]', '', ai_response)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Append the cleaned AI response to messages
        messages.append({"role": "assistant", "content": clean_text})
        return clean_text

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
        quit()
        
    except Exception as e:
        logger.debug(f"Error: Unexpected error occurred: {e}")
        return "Error: Unexpected error occurred. Please check your OpenAI API key."
# ----------------------------------------------------------------------------
    
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