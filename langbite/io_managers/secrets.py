from dotenv import load_dotenv
import os

def load_api_keys():
    load_dotenv()
    # Use .get() so a run does not crash when keys for providers you are NOT
    # using are absent (the previous os.environ[...] raised KeyError on the
    # first missing key, breaking even local GPT4ALL runs). Only the selected
    # model's provider key needs to be set.
    config = {
        'openai_api_key': os.environ.get("API_KEY_OPENAI", ""),
        'huggingface_api_key': os.environ.get("API_KEY_HUGGINGFACE", ""),
        'replicate_api_key': os.environ.get("API_KEY_REPLICATE", ""),
        'ollama_url': os.environ.get("OLLAMA_URL", "")
    }
    return config