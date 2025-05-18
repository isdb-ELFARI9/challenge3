import json
import os
from typing import Dict, List, Any, Optional, TypedDict
from fas_gaps_similarities_identifier_agent.data_models import State, FASAnalysisResult
from fas_gaps_similarities_identifier_agent.llm import get_llm, LLMProvider
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.write_to_file import write_to_file

# Make sure to import State, FASAnalysisResult, LLMProvider if they are used elsewhere or adjust as needed.
# from data_models import State, FASAnalysisResult # Assuming these are still relevant for other parts not shown
# from llm import LLMProvider # Assuming this is still relevant

class UpdatedFASDetail(TypedDict):
    """Details for a specific FAS recommended for update."""
    fas_id: str
    justification: str
    chain_of_thought: str
    referenced_gaps: List[str]
    referenced_similarities: List[str]

class NewFASDetail(TypedDict):
    """Details if a new FAS is deemed necessary."""
    justification: str
    chain_of_thought: str
    proposed_scope: str
    referenced_gaps_leading_to_new_fas: List[str]

class OverallVerdict(TypedDict):
    """The overall verdict from the synthesis."""
    fas_to_update: List[str]
    need_new_fas: bool
    overall_justification: str
    overall_chain_of_thought: str
    overall_referenced_gaps: List[str]
    overall_referenced_similarities: List[str]

class EnhancedSynthesisResult(TypedDict):
    """Type definition for the enhanced synthesis result, including detailed breakdowns."""
    overall_verdict: OverallVerdict
    updated_fas_details: List[UpdatedFASDetail]
    new_fas_details: Optional[NewFASDetail]

class SynthesisResult(TypedDict):
    """Type definition for synthesis result"""
    recommended_fas_updates: List[Dict[str, Any]]
    recommended_new_fas: Optional[Dict[str, Any]]
    justification: str
    chain_of_thought: str
    referenced_gaps: List[str]
    referenced_similarities: List[str]
import json
import os
# from data_models import State # Assuming State is defined and used as per original
# from llm import get_llm, LLMProvider # Assuming get_llm and LLMProvider are defined
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Ensure other necessary imports like State, get_llm, LLMProvider are correctly handled
# based on your project structure.

# (Your new TypedDict definitions from above would go here or be imported)

def synthesize_results(
    context: str,
    fas_results: Dict[str, Any], # Changed State to Any if State structure is not critical here
    llm_provider: str = "openai", # Assuming LLMProvider was a string or type alias
    output_file: str = "./synthesis_result.json"
) -> EnhancedSynthesisResult: # Changed return type
    """
    Synthesize the results from multiple FAS agents to provide a final detailed verdict.
    
    Args:
        context: The original user input/context.
        fas_results: Dictionary mapping FAS IDs to their corresponding final states or analysis results.
        llm_provider: The LLM provider to use.
        output_file: File to save the synthesis result.
        
    Returns:
        Enhanced synthesis result with detailed final verdict, justifications, and references.
    """
    print("\n--- Running Synthesis Agent ---")
    
    fas_analysis_summary = {}
    for fas_id, state_data in fas_results.items():
        # Adjust access to fas_analysis_result based on the actual structure of 'state_data'
        # For example, if state_data is the FASAnalysisResult itself:
        if isinstance(state_data, dict) and state_data.get("fas_analysis_result"):
             fas_analysis_summary[fas_id] = state_data["fas_analysis_result"]
        # Or if state_data itself is the analysis result directly:
        # fas_analysis_summary[fas_id] = state_data
        else:
            # Handle cases where the expected structure isn't found, if necessary
            fas_analysis_summary[fas_id] = {"error": "Analysis result not found or in unexpected format"}


    fas_results_str = json.dumps(fas_analysis_summary, indent=2)
    fas_results_str = fas_results_str.replace("{", "{{").replace("}", "}}")
    
    llm = get_llm(llm_provider, temperature=0.2)
    
    system_prompt = """You are an expert Financial Accounting Standards analyst for Islamic Finance.
Your task is to synthesize the results of multiple FAS (Financial Accounting Standards) analyses and provide a final detailed verdict.

Your output MUST be a valid JSON object with the following structure:
{{
  "overall_verdict": {{
    "fas_to_update": ["fas_x", "fas_y", ...],
    "need_new_fas": true|false,
    "overall_justification": "Comprehensive justification for the overall verdict.",
    "overall_chain_of_thought": "Detailed step-by-step reasoning process for arriving at the overall verdict.",
    "overall_referenced_gaps": ["key gap 1", ...],
    "overall_referenced_similarities": ["key similarity 1", ...]
  }},
  "updated_fas_details": [
    {{
      "fas_id": "fas_x",
      "justification": "Specific justification for why this particular FAS needs an update.",
      "chain_of_thought": "Reasoning process for concluding this FAS needs an update.",
      "referenced_gaps": ["gap specific to fas_x update"],
      "referenced_similarities": ["similarity specific to fas_x update"]
    }}
  ],
  "new_fas_details": {{ // This key should be present and populated if 'overall_verdict.need_new_fas' is true. Omit or set to null if false.
    "justification": "Justification for why a new FAS is needed.",
    "chain_of_thought": "Reasoning process for concluding a new FAS is required.",
    "proposed_scope": "A brief description of the potential scope or key areas the new FAS should cover.",
    "referenced_gaps_leading_to_new_fas": ["gap from analysis that strongly suggests the need for a new FAS"]
  }}
}}

Provide clear, concise, and well-reasoned justifications and chains of thought.
Ensure all referenced gaps and similarities are directly relevant.
If no FAS needs updating, "updated_fas_details" should be an empty list.
If 'overall_verdict.need_new_fas' is false, "new_fas_details" can be omitted or explicitly set to null.
"""
    
    human_prompt = f"""
ORIGINAL CONTEXT:
{context}

FAS ANALYSIS RESULTS:
{fas_results_str}

Based on the above analysis results, provide a final detailed verdict structured as per the JSON format specified in the system prompt.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    write_to_file("prompts.txt", "synthesizer prompt \n")
    write_to_file("prompts.txt", prompt_template)
    
    chain = prompt_template | llm | StrOutputParser()
    print("Invoking LLM for synthesis...")
    raw_synthesis_output = chain.invoke({})
    write_to_file("prompts.txt", "synthesizer output \n")
    write_to_file("prompts.txt", raw_synthesis_output)

    print(f"Raw Synthesis Output (first 300 chars): {raw_synthesis_output[:300]}...")
    
    try:
        cleaned_output = raw_synthesis_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()
        
        # Attempt to parse the potentially complex JSON
        parsed_json = json.loads(cleaned_output)

        # Basic validation for the new structure (can be expanded)
        if "overall_verdict" not in parsed_json:
            raise ValueError("Missing 'overall_verdict' in LLM output.")
        if "updated_fas_details" not in parsed_json:
             # It's okay for updated_fas_details to be empty, but the key should ideally exist
            parsed_json["updated_fas_details"] = []


        # Explicitly cast to EnhancedSynthesisResult for type checking, if your environment supports it well.
        # Otherwise, rely on the structure matching.
        synthesis_result: EnhancedSynthesisResult = parsed_json

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(synthesis_result, f, indent=2, ensure_ascii=False)
            print(f"Synthesis results saved to {output_file}")
        
        return synthesis_result
    
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse synthesis output as JSON: {e}. Output: {raw_synthesis_output[:500]}"
        print(f"ERROR: {error_msg}")
        # Consider how to handle this - perhaps return a default error structure or re-raise
        raise ValueError(error_msg) from e # Or a custom exception
    except Exception as e: # Catch other potential errors, like KeyError if structure is wrong
        error_msg = f"An unexpected error occurred during synthesis or processing the LLM output: {e}. Output: {raw_synthesis_output[:500]}"
        print(f"ERROR: {error_msg}")
        raise # Or a custom exception
    finally:
        print("--- Synthesis Agent Finished ---")

def format_synthesis_results(synthesis_result: EnhancedSynthesisResult) -> str:
    """
    Formats the enhanced synthesis results as a human-readable final verdict.
    It strictly includes the final verdict, justifications, chain of thoughts, and references.
    """
    if not synthesis_result:
        return "An error occurred: No synthesis result was provided."

    message = "# Final Verdict on FAS Analysis\n\n"

    overall_verdict = synthesis_result.get("overall_verdict")
    if not overall_verdict:
        return "Error: 'overall_verdict' is missing from the synthesis result. Cannot format output."

    # --- Overall Verdict Section ---
    message += "## Overall Verdict\n"
    fas_to_update = overall_verdict.get("fas_to_update", [])
    if fas_to_update:
        message += f"**FAS Standards to Update:** {', '.join(f.upper() for f in fas_to_update)}\n"
    else:
        message += "**FAS Standards to Update:** None identified for update.\n"

    need_new_fas = overall_verdict.get("need_new_fas", False)
    message += f"**Need for New FAS Standard:** {'Yes, a new FAS standard is indicated.' if need_new_fas else 'No, a new FAS standard is not indicated at this time.'}\n\n"

    message += "### Overall Justification:\n"
    message += f"{overall_verdict.get('overall_justification', 'No overall justification provided.')}\n\n"
    message += "### Overall Chain of Thought:\n"
    message += f"{overall_verdict.get('overall_chain_of_thought', 'No overall chain of thought provided.')}\n\n"

    overall_gaps = overall_verdict.get("overall_referenced_gaps", [])
    if overall_gaps:
        message += "#### Referenced Gaps (Supporting Overall Verdict):\n"
        for gap in overall_gaps:
            message += f"- {gap}\n"
        message += "\n"

    overall_similarities = overall_verdict.get("overall_referenced_similarities", [])
    if overall_similarities:
        message += "#### Referenced Similarities (Supporting Overall Verdict):\n"
        for sim in overall_similarities:
            message += f"- {sim}\n"
        message += "\n"

    # --- Details for FAS Updates Section ---
    updated_fas_details = synthesis_result.get("updated_fas_details", [])
    if updated_fas_details:
        message += "## Details for Recommended FAS Updates\n"
        for detail in updated_fas_details:
            fas_id = detail.get('fas_id', 'N/A').upper()
            message += f"\n### Update Details for {fas_id}:\n"
            message += f"**Justification for Update:** {detail.get('justification', 'N/A')}\n"
            message += f"**Chain of Thought for Update Decision:** {detail.get('chain_of_thought', 'N/A')}\n"
            
            gaps = detail.get("referenced_gaps", [])
            if gaps:
                message += "**Referenced Gaps Specific to this Update:**\n"
                for gap in gaps:
                    message += f"  - {gap}\n"
            
            similarities = detail.get("referenced_similarities", [])
            if similarities:
                message += "**Referenced Similarities Specific to this Update:**\n"
                for sim in similarities:
                    message += f"  - {sim}\n"
            message += "\n" # Add a newline for separation before the next FAS detail

    # --- Details for New FAS Requirement Section ---
    new_fas_details = synthesis_result.get("new_fas_details")
    # Only show this section if need_new_fas is true AND details are provided
    if need_new_fas and new_fas_details:
        message += "## Details Regarding the Need for a New FAS Standard\n"
        message += f"**Justification for New FAS:** {new_fas_details.get('justification', 'N/A')}\n"
        message += f"**Chain of Thought for New FAS Decision:** {new_fas_details.get('chain_of_thought', 'N/A')}\n"
        message += f"**Proposed Scope for New FAS:** {new_fas_details.get('proposed_scope', 'N/A')}\n"
        
        gaps_new_fas = new_fas_details.get("referenced_gaps_leading_to_new_fas", [])
        if gaps_new_fas:
            message += "**Referenced Gaps Influencing New FAS Decision:**\n"
            for gap in gaps_new_fas:
                message += f"  - {gap}\n"
        message += "\n"
    elif need_new_fas and not new_fas_details:
        message += "## Details Regarding the Need for a New FAS Standard\n"
        message += "The overall verdict indicates a new FAS is needed, but specific details (justification, scope, etc.) were not provided in the 'new_fas_details' section of the raw output.\n\n"
        
    return message