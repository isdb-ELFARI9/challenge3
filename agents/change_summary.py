from utils.llm import call_gemini_llm
import json

def build_change_summary_prompt(change_log, reasoning_trace, user_context) -> str:
    return (
        "You are the Change Summary Agent for an Islamic finance standards review system.\n\n"
        "Your job is to write a clear, concise summary for the user, explaining what was changed in the FAS, which parts were updated, and why.\n\n"
        "Use the change log and reasoning trace to identify the most important changes and their rationale.\n"
        "Write in plain, user-friendly language.\n\n"
        f"User context: {user_context}\n\n"
        f"Change log: {change_log}\n\n"
        f"Reasoning trace: {reasoning_trace}\n\n"
        "Respond ONLY with the summary as a plain text string."
    )

def change_summary_agent(change_log, reasoning_trace, user_context) -> str:
    prompt = build_change_summary_prompt(change_log, reasoning_trace, user_context)
    response = call_gemini_llm(prompt)
    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    return response 