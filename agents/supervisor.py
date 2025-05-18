from utils.llm import call_gemini_llm
from utils.write_to_file import write_to_file


def supervisor_agent_llm(proposals):
    # Compose prompt string with proposals summarized
    prompt = "You are a supervisor agent tasked with evaluating these proposals for updating FAS 4.\n\n"
    for i, p in enumerate(proposals, 1):
        llm = p["llm"]
        content = p["content"]

        prompt += f"Proposal {i} by {llm}:\n"
        prompt += f"Shariah Solution: {content['shariah_solution']}\n"
        prompt += f"Accounting Rationale: {content['accounting_rationale']}\n"
        prompt += f"Updated Shariah Clauses:\n"
        for clause in content['updated_shariah_clauses']:
            prompt += f" - [{clause['clause_id']}] {clause['text']} (ref: {clause['reference']})\n"
        prompt += f"Updated Accounting Clauses:\n"
        for clause in content['updated_accounting_clauses']:
            prompt += f" - [{clause['clause_id']}] {clause['text']} (ref: {clause['reference']})\n"
        prompt += f"References: {', '.join(content['references'])}\n\n"

    prompt += (
        "Your task:\n"
        "- Select the best proposals that do not conflict.\n"
        "- Provide reasoning.\n"
        "- Suggest merged final clauses if applicable.\n"
        "- Include both Shariah and accounting clauses from the selected proposals.\n"
        "- Merge clauses where appropriate. Preserve clause IDs and their content.\n"
        "- If references are included in the proposals, incorporate them into the merged clauses where relevant.\n"
        "- Provide a detailed justification for your selection and merging decisions, referring to specific elements from the proposals (including clause IDs, content, and references).\n"
        "Please respond in JSON format with the following fields:\n"
        "{\n"
        "  'selected_proposals': [proposal_numbers],\n"
        "  'reasoning': str,\n"
        "  'merged_shariah_clauses': {\n"
        "    'clause_id': 'text',\n"
        "    ...\n"
        "  },\n"
        "  'merged_accounting_clauses': {\n"
        "    'clause_id': 'text',\n"
        "    ...\n"
        "  }\n"
        "}\n"
        "Include both Shariah and accounting clauses from the selected proposals.\n"
        "Merge clauses where appropriate. Preserve clause IDs and their content.\n"

    )

    # Call your preferred LLM (GPT or Gemini) for supervisor
    write_to_file("prompts.txt", "supervisor agent prompt\n")
    write_to_file("prompts.txt", prompt)
    response_text = call_gemini_llm(prompt)  # or call_gemini_llm(prompt)
    write_to_file("prompts.txt", "supervisor agent response\n")
    write_to_file("prompts.txt", response_text)


    # Try to parse JSON from response_text
    import json
    try:
        # must exlude ``` json and ``` if they exist first
        response_text = response_text.replace("```json", "").replace("```", "")
        print("response_text after removing ```json and ```")
        return json.loads(response_text)
    except json.JSONDecodeError:
        # fallback - return raw response for manual inspection
        return {"raw_response": response_text}
