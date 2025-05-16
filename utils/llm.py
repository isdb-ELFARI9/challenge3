import os
from typing import Dict, Any
import openai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_gemini_llm(prompt: str) -> str:
    """
    Get response from Google's Gemini model.
    
    Args:
        prompt (str): The prompt to send to the model
        
    Returns:
        str: The response from the model
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-pro")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model_name)
    response = llm.generate_content(prompt)
    return response.text

def get_llm_response(prompt: str) -> Dict[str, Any]:
    """
    Get response from OpenAI's GPT model.
    
    Args:
        prompt (str): The prompt to send to the model
        
    Returns:
        Dict[str, Any]: The structured response from the model
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or your preferred model
            messages=[
                {"role": "system", "content": "You are an expert in Islamic finance and accounting standards. Analyze the differences between FAS versions and provide detailed, structured responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        
        # The response should be in a structured format that can be parsed
        # You might need to adjust this based on your specific needs
        try:
            import json
            return json.loads(content)
        except json.JSONDecodeError:
            # If the response isn't valid JSON, return it as a string
            return {"raw_response": content}
            
    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return {"error": str(e)}