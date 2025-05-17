
from fas_gaps_similarities_identifier_agent.config import supported_fas_list
import os
import json
from pinecone import Pinecone
import openai
from langchain_core.prompts import ChatPromptTemplate
from fas_gaps_similarities_identifier_agent.llm import get_llm, LLMProvider
from langchain_core.output_parsers import StrOutputParser
from fas_gaps_similarities_identifier_agent.data_models import State, FASAnalysisResult
from typing import TypedDict, List, Dict, Any, Optional
from fas_gaps_similarities_identifier_agent.fas_retriever_agent import FASRetriever


def fas_gaps_and_similarities_detector_agent(state: State, target_fas_id: str = "fas_4", llm_provider: LLMProvider = "gemini", 
              output_file: str = "./out.json") -> State: # Added output_file parameter
    """
    Agent that will extract the gaps and similarities between the provided context and the target FAS.
    The output will be a structured JSON with a list of the gaps and similarities,
    each element will have a justification, used references, chain of thoughts,
    and an overall score of similarity.
    Modifies the state with the analysis result.
    
    Args:
        state: Current state dictionary
        target_fas_id: ID of the target FAS to analyze against
        llm_provider: Provider for the LLM to use
        output_file: Optional path to save the results (default: None)
    """
    print(f"\n--- Running FAS Agent ---")
    
    # Extract context from the messages in state
    context = ""
    if state.get("messages") and len(state["messages"]) > 0:
        # Typically extract context from the latest user message
        for message in reversed(state["messages"]):
            if message.get("role") == "human" and message.get("content"):
                context = message["content"]
                break
    
    print(f"Context: {context[:100]}...")
    print(f"Target FAS ID: {target_fas_id}")

    #  Retrieve knowledge using RAG
    retriever = FASRetriever()

    results = retriever.retrieve(query=target_fas_id)
    filtered_results = [doc for doc in results if target_fas_id.upper() in doc.id]
    knowledge = "["
    for doc in filtered_results:
        knowledge += f"{'{'} Document_ID: {doc.id},\n"
        knowledge += f"Document_Content: {doc.text}, \n"
        knowledge += f"Document_Metadata: {doc.metadata} {'},'}\n\n"
    knowledge += "]"

    
    # Generate the system prompt correctly
    system_message_content = get_system_prompt(
        target_fas_id,
        knowledge
    )

    llm = get_llm(llm_provider)

    #  Construct the prompt for the LLM
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message_content),
        ("human", context) # The actual context is passed here
    ])

    # Run the LLM chain
    chain = prompt_template | llm | StrOutputParser()
    print("Invoking LLM for FAS analysis...")
    raw_llm_output = chain.invoke({"context": context}) # Pass context as input variable
    print(f"Raw LLM Output (first 200 chars): {raw_llm_output[:200]}...")

    #  Parse the output and update state
    if "thoughts" not in state:
        state["thoughts"] = []
    state["thoughts"].append(f"Raw LLM output for FAS analysis received. Attempting to parse JSON.") # Log the attempt

    parsed_analysis: Optional[FASAnalysisResult] = None
    try:
        # It's good practice to clean the output if LLMs sometimes add ```json ... ```
        cleaned_output = raw_llm_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()

        parsed_analysis = json.loads(cleaned_output)
        parsed_analysis["target_fas_id"] = target_fas_id
        state["fas_analysis_result"] = parsed_analysis

        print("Successfully parsed LLM output into JSON.")
        state["thoughts"].append("Successfully parsed FAS analysis JSON.")
        
        # Save results to file if output_file is specified
        if output_file:
            try:
                # Save both raw output and parsed JSON
                save_results_to_file(output_file, 
                                    raw_llm_output=raw_llm_output, 
                                    parsed_json=parsed_analysis,
                                    target_fas_id=target_fas_id,
                                    context_summary=context[:200])
                print(f"Results saved to {output_file}")
                state["thoughts"].append(f"FAS analysis results saved to {output_file}")
            except Exception as e:
                error_msg = f"Failed to save results to file: {e}"
                print(f"WARNING: {error_msg}")
                state["thoughts"].append(error_msg)
                
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM output as JSON: {e}. Output: {raw_llm_output[:500]}"
        print(f"ERROR: {error_msg}")
        state["thoughts"].append(f"FAS Agent Error: {error_msg}")
        state["fas_analysis_result"] = None # Set to None on error
    except Exception as e:
        error_msg = f"An unexpected error occurred during FAS analysis processing: {e}"
        print(f"ERROR: {error_msg}")
        state["thoughts"].append(f"FAS Agent Error: {error_msg}")
        state["fas_analysis_result"] = None

    # Update other relevant state fields
    state["current_target_fas_id"] = target_fas_id
    state["current_context"] = context
    # state["completed"] = True # Or based on broader workflow logic

    print("--- FAS Agent Finished ---")
    return state


def save_results_to_file(output_file: str, 
                         raw_llm_output: str, 
                         parsed_json: dict,
                         target_fas_id: str,
                         context_summary: str) -> None:
    """
    Save the FAS analysis results to a file.
    
    Args:
        output_file: Path to save the results
        raw_llm_output: Raw output from the LLM
        parsed_json: Parsed JSON result
        target_fas_id: The target FAS ID
        context_summary: Summary of the context
    """
    import json
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # f.write(f"# FAS Analysis Results\n")
        # f.write(f"Generated: {timestamp}\n")
        # f.write(f"Target FAS ID: {target_fas_id}\n")
        # f.write(f"Context Summary: {context_summary}...\n\n")
        
        # f.write("## Parsed JSON Result\n")
        # f.write("```json\n")
        json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        # f.write("\n```\n\n")
        
        # f.write("## Raw LLM Output\n")
        # f.write("```\n")
        # f.write(raw_llm_output)
        # f.write("\n```\n")
    
    return

def get_system_prompt(target_fas_id: str, fas_knowledge: str) -> str:
    """
    Generates the system prompt for the FAS (Financial Accounting Standards) analysis agent.

    Args:
        target_fas_id: The identifier of the target AAOIFI FAS standard to be analyzed.
        fas_knowledge: The FAS related knowledge collected from RAG (Retrieval Augmented Generation)
                       relevant to the analysis.

    Returns:
        A string representing the system prompt for the LLM agent.
    """

    # Define the expected JSON structure for clarity in the prompt
    # Note: Double curly braces to escape them in the f-string
    json_structure_example = """
{{
  "analysis_summary": {{
    "overall_assessment": "A high-level statement describing the overall relationship between the provided context and the target FAS, highlighting key findings.",
    "key_metrics": {{
      "gaps_identified_count": 2,
      "similarities_identified_count": 3
    }}
  }},
  "identified_gaps": [
    {{
      "description": "Concise description of the identified gap (e.g., 'Context introduces a requirement for digital asset accounting not covered in FAS X').",
      "justification": "Detailed reasoning explaining why this is a gap, referencing specific aspects of the context and the absence of corresponding guidance in the FAS knowledge.",
      "references": [
        "Relevant quote or paraphrase from user-provided context causing the gap",
        "Statement indicating lack of coverage in provided FAS knowledge or standard (e.g., 'FAS knowledge section Y does not address Z')"
      ],
      "chain_of_thought": "My step-by-step reasoning: 1. Analyzed context for new financial implications. 2. Compared against FAS knowledge. 3. Noticed absence of specific guidance. 4. Concluded it's a gap.",
      "score": 0.8
    }}
  ],
  "identified_similarities": [
    {{
      "description": "Concise description of the identified similarity (e.g., 'Context's emphasis on ethical screening aligns with FAS X, para Y').",
      "justification": "Detailed reasoning explaining why this is a similarity, referencing specific aspects of the context and corresponding guidance in the FAS knowledge.",
      "references": [
        "Relevant quote or paraphrase from user-provided context",
        "Specific quote or section from FAS knowledge/standard (e.g., 'FAS X, Paragraph Y.Z states...')",
      ],
      "chain_of_thought": "My step-by-step reasoning: 1. Analyzed context for financial principles. 2. Compared with FAS knowledge. 3. Found explicit alignment. 4. Concluded it's a similarity.",
      "score": 0.9
    }}
  ]
}}
"""

    system_prompt = f"""
You are an expert Financial Accounting Standards (FAS) Analysis Agent, specializing in AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions) standards.
Your primary mission is to meticulously analyze a given `context` (which will be provided in the user's message) against a specific AAOIFI FAS standard, identified by `target_fas_id`, and relevant `fas_knowledge` (provided below).

Your goal is to identify:
1.  **Gaps**: Aspects, requirements, scenarios, or financial implications presented in the `context` that are NOT adequately addressed, covered, or are contradicted by the `target_fas_id` and the provided `fas_knowledge`.
2.  **Similarities**: Aspects, requirements, scenarios, or financial principles presented in the `context` that ARE aligned with, covered by, or reinforced by the `target_fas_id` and the provided `fas_knowledge`.

**Target FAS Standard for this Analysis:** {target_fas_id}

**Relevant FAS Knowledge (from RAG):**
{fas_knowledge}

**Instructions for Analysis and Output:**

1.  **Analyze Thoroughly**: Carefully read and understand the user-provided `context`. Compare it critically against the `target_fas_id` and the `fas_knowledge`.
2.  **Identify Key Elements**: For each identified gap or similarity, you must provide:
    *   `description`: A clear and concise summary of what the gap or similarity is.
    *   `justification`: Detailed reasoning for your conclusion. Explain *why* it's a gap or a similarity, linking parts of the `context` to the (lack of) corresponding parts in the `fas_knowledge` or general understanding of the `target_fas_id`.
    *   `references`: Specific references that support your finding. These can be direct quotes or paraphrased summaries from the `user-provided context` and the `fas_knowledge`. If a direct reference from FAS knowledge is about its absence, state that (e.g., "No specific guidance found in FAS knowledge regarding X mentioned in context").
    *   `chain_of_thought`: A brief, step-by-step outline of your thought process for identifying and categorizing the item.
    *   `score`: A numerical score between 0.0 and 1.0.
        *   For **gaps**: This score represents the significance or potential impact of the gap (0.0 = minor/irrelevant, 1.0 = very significant/critical).
        *   For **similarities**: This score represents the degree of alignment or strength of the similarity (0.0 = weak/tenuous, 1.0 = very strong/direct alignment).

3.  **Structured JSON Output**: You MUST format your entire response as a single, valid JSON object. Do NOT include any text or explanations outside of this JSON object.
    The JSON object must strictly adhere to the following structure (see example below for content guidance):

{json_structure_example}

Important Considerations:
The user-provided context will be supplied in the upcoming user message.
If the fas_knowledge is insufficient to make a definitive judgment on a point from the context, clearly state this in your justification for that item.
If no gaps or no similarities are found, return an empty list [] for the respective key (identified_gaps or identified_similarities).
Be precise, analytical, and objective in your assessment.
Ensure your output is ONLY the JSON object. No introductory phrases like "Here is the JSON:" or concluding remarks.
Begin your analysis once you receive the user's message containing the context.
"""
    return system_prompt.strip()