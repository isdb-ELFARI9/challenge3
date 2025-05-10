from schemas.uiria import UIRIAInput, UIRIAOutput
from utils.llm import call_gemini_llm
import json

def build_uiria_prompt(user_prompt: str) -> str:
    return (
        f"You are the User Interface & Request Intake Agent (UIRIA) for an Islamic finance standards review system.\n\n"
        f"Your job is to:\n"
        f"- Read the user's raw input (which may be messy, unstructured, or incomplete).\n"
        f"- Extract the main context, identify the single most relevant FAS (Financial Accounting Standard), extract key entities (products, terms, issues), and infer the user's intent.\n"
        f"- Output your findings in a structured JSON format with the following fields:\n"
        f"  - context: a concise summary of the user's situation or problem.\n"
        f"  - identified_FAS: the single most relevant FAS number or name mentioned or implied (as a string).\n"
        f"  - extracted_entities: a list of key terms, products, or issues.\n"
        f"  - user_intent: a short phrase describing what the user wants (e.g., 'clarification', 'update', 'gap analysis').\n\n"
        f"Example output:\n"
        f"{{\n"
        f"  \"context\": \"...\",\n"
        f"  \"identified_FAS\": \"FAS 4\",\n"
        f"  \"extracted_entities\": [\"diminishing Musharaka\", \"shirkah al-ʿaqd\", \"REITs\"],\n"
        f"  \"user_intent\": \"clarification and update\"\n"
        f"}}\n\n"
        f"User input:\n\"\"\"{user_prompt}\"\"\"\n\n"
        f"Respond ONLY with the JSON object."
    )

def uiria_agent(user_prompt: str) -> UIRIAOutput:
    prompt = build_uiria_prompt(user_prompt)
    response = call_gemini_llm(prompt)
    response = response.strip()
    if response.startswith("```"):
        # Remove the first line (``` or ```json)
        response = response.split('\n', 1)[1]
        # Remove the last line if it's a closing code block
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    data = json.loads(response)
    return UIRIAOutput(**data)