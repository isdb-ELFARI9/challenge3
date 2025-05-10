import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

def call_gemini_llm(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-pro")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model_name)
    response = llm.generate_content(prompt)
    return response.text