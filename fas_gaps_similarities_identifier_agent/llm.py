import os
from typing import  Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

# Load environment variables
load_dotenv()

# Define available LLM providers
LLMProvider = Literal["openai", "gemini"]

# Configure API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Function to get the appropriate LLM based on provider
def get_llm(provider: LLMProvider = "openai", temperature: float = 0.7) -> BaseChatModel:
    """Get the appropriate language model based on the provider."""
    if provider == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=temperature
        )
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model="gemini-pro",
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")