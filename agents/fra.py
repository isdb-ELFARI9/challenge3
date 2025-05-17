import os
import json
from typing import List, Dict, Any, Optional

# Assume these exist and are configured globally or passed in
from utils.llm import call_gemini_llm # Your function to call the Gemini LLM
from utils.db import get_pipeline_run # Function to retrieve past run data from SQLite
from utils.fas_utils import get_fas_namespace # Utility to get FAS namespace

# Pinecone and OpenAI imports and setup (Centralized setup is better in a real app)
from pinecone import Pinecone
import openai
from pydantic import BaseModel, Field, ValidationError # Use Pydantic for schema validation

# --- Environment Variable Loading and Configuration ---
# Assuming dotenv was loaded in the main app, but check here defensively
# from dotenv import load_dotenv
# load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_FAS_NAME = os.getenv("PINECONE_INDEX_FAS")
PINECONE_INDEX_SS_NAME = os.getenv("PINECONE_INDEX_SS")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for required environment variables (optional here if done in main, but safe)
required_vars = [
    PINECONE_API_KEY, PINECONE_ENVIRONMENT,
    PINECONE_INDEX_FAS_NAME, PINECONE_INDEX_SS_NAME,
    OPENAI_API_KEY
]
if any(v is None for v in required_vars):
     # In a service, might log and raise or handle differently
     print("Warning: One or more required environment variables for Pinecone/OpenAI are not set. RAG/Embeddings may not work.")
     # For a PoC, we might proceed but RAG calls will fail

# Initialize Pinecone client (new SDK) - Ensure this is only done once if possible in a real app
pc = None
index_fas = None
index_ss = None
try:
    if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        if PINECONE_INDEX_FAS_NAME in pc.list_indexes():
             index_fas = pc.Index(PINECONE_INDEX_FAS_NAME)
        else:
             print(f"Warning: FAS index '{PINECONE_INDEX_FAS_NAME}' not found.")
        if PINECONE_INDEX_SS_NAME in pc.list_indexes():
             index_ss = pc.Index(PINECONE_INDEX_SS_NAME)
        else:
             print(f"Warning: SS index '{PINECONE_INDEX_SS_NAME}' not found.")
    else:
        print("Warning: Pinecone environment variables not fully set. Pinecone client not initialized.")

except Exception as e:
    print(f"Warning: Could not initialize Pinecone. RAG will not work. {e}")
    index_fas = None
    index_ss = None


# Embedding function using OpenAI - Ensure this is only done once if possible
openai.api_key = OPENAI_API_KEY
def get_embedding(text: str) -> list:
    if not openai.api_key:
        print("OpenAI API key not set. Cannot generate embeddings.")
        return [] # Handle case where key is missing
    try:
        # Replace newline characters which can cause issues in embeddings
        text = text.replace('\n', ' ')
        response = openai.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding failed: {e}")
        return []

# --- RAG retrieval functions ---
# Adapt from your existing ones, adding checks for initialized indexes
def retrieve_knowledge_from_pinecone_fas(query: str, fas_namespace: str) -> str:
    if not index_fas or not fas_namespace:
        print("FAS index or namespace not available. Skipping FAS RAG.")
        return "[FAS RAG search skipped]"
    embedding = get_embedding(query)
    if not embedding:
        return "[FAS RAG search skipped due to embedding failure]"
    try:
        # Query Pinecone for the top 3 matches in the correct namespace
        result = index_fas.query(vector=embedding, top_k=3, namespace=fas_namespace, include_metadata=True) # Increased top_k for better context
        if result and result.matches:
            # Combine top results for better context, add score threshold
            relevant_docs = [match.metadata.get('text', '') for match in result.matches if match.score > 0.7]
            if relevant_docs:
                 return "Relevant FAS Knowledge:\n---\n" + "\n---\n".join(relevant_docs)
            return "[No highly relevant FAS document found via RAG]"
        return "[No relevant FAS document found via RAG]"
    except Exception as e:
        print(f"FAS RAG search failed: {e}")
        return "[FAS RAG search failed]"


def retrieve_knowledge_from_pinecone_ss(query: str, ss_namespace: str) -> str:
    if not index_ss or not ss_namespace:
        print("SS index or namespace not available. Skipping SS RAG.")
        return "[SS RAG search skipped]"
    embedding = get_embedding(query)
    if not embedding:
        return "[SS RAG search skipped due to embedding failure]"
    try:
        # Query Pinecone for the top 3 matches
        result = index_ss.query(vector=embedding, top_k=3, namespace=ss_namespace, include_metadata=True) # Increased top_k
        if result and result.matches:
             # Combine top results for better context, add score threshold
            relevant_docs = [match.metadata.get('text', '') for match in result.matches if match.score > 0.7]
            if relevant_docs:
                 return "Relevant Shariah Standards Knowledge:\n---\n" + "\n---\n".join(relevant_docs)
            return "[No highly relevant SS document found via RAG]"
        return "[No relevant SS document found via RAG]"
    except Exception as e:
        print(f"SS RAG search failed: {e}")
        return "[SS RAG search failed]"

# --- Pydantic Schemas for FRA ---

# Schema for the specific feedback item provided by the manager
class ManagerFeedbackItem(BaseModel):
    change_id: str = Field(..., description="Identifier for the specific change being reviewed (e.g., index in changes list, or generated ID).")
    reason: str = Field(..., description="Detailed reason for the feedback (e.g., 'Justification weak', 'New text seems incorrect Shariah-wise').")

# Input schema for the Refinement Agent
class FRAInput(BaseModel):
    run_id: int = Field(..., description="The ID of the pipeline run data to retrieve from the database.")
    feedback: List[ManagerFeedbackItem] = Field(..., description="List of specific feedback items from the manager.")
    # We don't need to pass knowledge_indexes here, the agent will retrieve from the stored data

# Schema for the outcome of reviewing a single change
class ReviewedChangeOutcome(BaseModel):
    change_id: str = Field(..., description="The ID of the change that was reviewed.")
    status: str = Field(..., description="Outcome of the review: 'updated_justification' | 'updated_text' | 'discarded' | 'requires_manual_review'")
    revised_change_object: Optional[Dict[str, Any]] = Field(None, description="The revised change object if status is 'updated_justification' or 'updated_text'.")
    feedback_analysis: str = Field(..., description="Analysis of the feedback and decision rationale based on the trace.")
    justification_improvement_notes: Optional[str] = Field(None, description="Specific notes on how justification was improved, citing sources from trace.")
    text_improvement_notes: Optional[str] = Field(None, description="Specific notes on how new_text was improved, citing reasoning from trace.")


# Output schema for the Refinement Agent
class FRAOutput(BaseModel):
    reviewed_changes_outcome: List[ReviewedChangeOutcome]
    summary: str = Field(..., description="Overall summary of the refinement process outcomes.")

# --- FRA Prompt Builder ---

def build_fra_prompt(
    original_context: str,
    fas_number: str,
    original_fas_markdown: str, # Need original FAS text to understand change context fully
    full_reasoning_trace: Dict[str, Any],
    fas_diff_output: Dict[str, Any],
    feedback: List[ManagerFeedbackItem],
    fas_knowledge: str, # RAG results for relevant FAS
    ss_knowledge: str # RAG results for relevant SS
) -> str:
    """Builds the prompt for the Refinement Agent."""

    # Format feedback clearly for the LLM
    formatted_feedback = "\n".join([
        f"- Change ID: {item.change_id}\n  Reason: {item.reason}" for item in feedback
    ])

    # Find the specific changes being referenced in the feedback
    # This requires mapping change_id from feedback to the actual change object in fas_diff_output
    # For simplicity, assuming change_id might be the index or a unique identifier you add
    changes_to_review = []
    all_changes_list = fas_diff_output.get("changes", [])
    for feedback_item in feedback:
        # Simple lookup by change_id string match (assuming change_id is unique string)
        # A more robust approach might involve a hash or index + section_id
        found_change = None
        for change_obj in all_changes_list:
             # Need a reliable way to match feedback_item.change_id to a change_obj
             # If change_id is the index from fas_diff_output['changes'], use index
             # If change_id is a unique ID you generated, match that ID
             # For this PoC, let's assume change_id is a string that can be found in the change object itself or metadata
             # A safer PoC approach is to pass index from feedback
             # Let's revert to the index assumption for the PoC prompt structure
             try:
                 # Assuming change_id is a string like "change_at_index_X" or similar
                 index = int(feedback_item.change_id.split('_')[-1]) # Example: "change_at_index_3" -> 3
                 if 0 <= index < len(all_changes_list):
                      found_change = all_changes_list[index]
                      break # Found the change
                 else:
                     print(f"Warning: Change ID index out of bounds: {feedback_item.change_id}")
             except (ValueError, IndexError):
                 print(f"Warning: Could not parse index from Change ID: {feedback_item.change_id}")
                 # Fallback or error handling if index parsing fails

        if found_change:
             changes_to_review.append({
                 "feedback_id": feedback_item.change_id,
                 "reason": feedback_item.reason,
                 "change_object": found_change
             })
        else:
             changes_to_review.append({
                 "feedback_id": feedback_item.change_id,
                 "reason": feedback_item.reason,
                 "change_object": None, # Indicate change not found
                 "error": "Change ID not found or processed in fas_diff output"
             })


    formatted_changes_to_review = json.dumps(changes_to_review, indent=2)

    # --- The Refinement Agent Prompt ---
    # Escaped curly braces {{ and }} for literal JSON parts
    return f"""You are the FAS Refinement Agent (FRA). Your role is to review specific proposed changes to an Islamic Financial Accounting Standard (FAS) based on manager feedback and the comprehensive analysis performed by previous agents in a pipeline. Your goal is to improve the proposed changes or their justifications using the provided context and reasoning trace.

        Inputs Provided:
        - Original User Context: {{original_context}}
        - Target FAS: {{fas_number}}
        - Original FAS Text (Relevant Sections):
        ```markdown
        {{original_fas_markdown}}
        ```
        - Full Multi-Agent Reasoning Trace (includes outputs from all agents: FCIA, SPIA, ARDA, STSA, DocumentComposer, ChangeSummary, and FAS_Diff):
        ```json
        {{json.dumps(full_reasoning_trace, indent=2)}}
        ```
        - Complete FAS_Diff Output (details all proposed changes):
        ```json
        {{json.dumps(fas_diff_output, indent=2)}}
        ```
        - Specific Manager Feedback (identifying changes to review and reasons):
        ```json
        {{formatted_changes_to_review}}
        ```
        - Relevant Knowledge from FAS Index (RAG Results):
        {{fas_knowledge}}
        - Relevant Knowledge from Shariah Standards Index (RAG Results):
        {{ss_knowledge}}

        Your Task:
        For *each* item listed in the "Specific Manager Feedback":
        1. Identify the specific change object it refers to from the "Complete FAS_Diff Output". If a change ID cannot be found or processed, report it in the `feedback_analysis`.
        2. Carefully analyze the "Reason" provided in the feedback for that change.
        3. Review the "Full Multi-Agent Reasoning Trace" to understand *why* the original change and its justification were proposed. Pay close attention to the rationales from SPIA, ARDA, and the change descriptions from STSA. Look for details, Shariah principles, accounting logic, and sources mentioned in the trace that are relevant to this specific change.
        4. Determine the best way to address the feedback:
            - **Improve Justification:** If the underlying reasoning in the trace is solid but the original justification in the FAS_Diff output was poorly written or incomplete for *this specific text change*. Extract relevant details and sources *from the trace* to write a more convincing justification for the *existing new_text*. Update the `justification` field in the `revised_change_object`.
            - **Improve New Text:** If the feedback suggests the *content* of the `new_text` itself is problematic, review the trace to see if there's alternative reasoning or a different interpretation that suggests a revised `new_text`. This is more complex and might require careful re-interpretation of SPIA/ARDA outputs in light of RAG results if applicable. Update the `new_text` field in the `revised_change_object`.
            - **Discard Change:** If the underlying reasoning in the trace for this change is weak, illogical, or contradicts fundamental principles found in the RAG results, and you cannot find a strong basis for it in the trace. Set `status` to 'discarded' and `revised_change_object` to null.
            - **Requires Manual Review:** If the feedback points to a deep issue that cannot be resolved by just re-synthesizing from the trace or requires external input. Set `status` to 'requires_manual_review' and `revised_change_object` to null.

        5. Output a JSON array containing the outcome for *each* change reviewed, using the `reviewed_changes_outcome` structure below. For each outcome:
            - Provide the `change_id`.
            - Set the `status` ('updated_justification', 'updated_text', 'discarded', 'requires_manual_review').
            - If status is 'updated_justification' or 'updated_text', include the `revised_change_object` (the original change object with the improved justification and/or `new_text`).
            - Provide `feedback_analysis` explaining your reasoning and decision.
            - If justification/text was improved, include notes in `justification_improvement_notes` or `text_improvement_notes` mentioning what was added/changed and *which part of the trace or RAG results* supported it.

        **Output JSON Format:**
        ```json
        {{
        "reviewed_changes_outcome": [
            {{
            "change_id": "...",
            "status": "updated_justification" | "updated_text" | "discarded" | "requires_manual_review",
            "revised_change_object": {{
                "old_text": "...",
                "new_text": "...", // Could be the original new_text or revised
                "justification": "...", // Could be the original justification or revised
                "section_id": "...",
                "change_type": "..."
            }} | null, // Null if discarded or requires manual review
            "feedback_analysis": "Analysis of feedback and decision.",
            "justification_improvement_notes": "Notes on justification changes...", // If applicable
            "text_improvement_notes": "Notes on new_text changes...", // If applicable
            }},
            // ... outcomes for other reviewed changes
        ],
        "summary": "Overall summary of the refinement process."
        }}

        Respond ONLY with the JSON object.
        """

# --- Refinement Agent Function ---

def fra_agent(fra_input: FRAInput) -> FRAOutput:
    """
    Refinement Agent that reviews specific changes based on feedback and trace.
    Retrieves pipeline run data, performs RAG, prompts LLM, and validates output.
    """
    print(f"FRA: Starting refinement for run ID {fra_input.run_id} with {len(fra_input.feedback)} feedback items.")

    # 1. Retrieve pipeline run data
    run_data = get_pipeline_run(fra_input.run_id)
    if not run_data:
        print(f"FRA Error: Pipeline run with ID {fra_input.run_id} not found.")
        # Return an error outcome or raise exception
        return FRAOutput(
            reviewed_changes_outcome=[],
            summary=f"Error: Pipeline run with ID {fra_input.run_id} not found."
        )

    original_context = run_data.get("user_prompt", "N/A") # Use get with default
    fas_number = run_data.get("fas_number", "N/A")
    full_reasoning_trace = run_data.get("reasoning_trace", {})
    original_fas_markdown = run_data.get("original_fas_markdown_text", "Original FAS text not stored or found.")
    fas_diff_output = full_reasoning_trace.get("fas_diff", {"changes": []})

    # 2. Perform RAG based on the general context and specific feedback areas
    # We can query using the original context + summarized feedback
    feedback_reasons = [item.reason for item in fra_input.feedback]
    rag_query = f"Context: {original_context}\nFeedback topics: {', '.join(feedback_reasons)}"
    print(f"FRA: RAG query: {rag_query[:100]}...") # Print snippet of query

    fas_namespace = None
    if isinstance(fas_number, str) and fas_number.lower().startswith('fas'):
         fas_namespace = get_fas_namespace(fas_number)
    elif isinstance(fas_number, list) and fas_number:
         # For PoC, use the first FAS number's namespace if it's a list
         fas_namespace = get_fas_namespace(fas_number[0])
    # If fas_number is not in expected format, fas_namespace remains None

    # Retrieve relevant FAS knowledge
    fas_knowledge = retrieve_knowledge_from_pinecone_fas(rag_query, fas_namespace)
    print(f"FRA: Retrieved FAS knowledge (first 100 chars): {fas_knowledge[:100]}...")

    # Retrieve relevant SS knowledge
    # A better approach might be to look at SS references in the reasoning trace
    # For PoC, query a default SS namespace or try to infer from context/trace
    # Let's query using the context and feedback reasons, assuming SS12 is a common relevant standard
    ss_knowledge = retrieve_knowledge_from_pinecone_ss(rag_query, "SS12") # Example SS namespace
    print(f"FRA: Retrieved SS knowledge (first 100 chars): {ss_knowledge[:100]}...")


    # 3. Build the prompt for the LLM
    prompt = build_fra_prompt(
        original_context=original_context,
        fas_number=fas_number,
        original_fas_markdown=original_fas_markdown,
        full_reasoning_trace=full_reasoning_trace,
        fas_diff_output=fas_diff_output,
        feedback=fra_input.feedback,
        fas_knowledge=fas_knowledge,
        ss_knowledge=ss_knowledge
    )
    # print(f"FRA: Generated prompt (first 500 chars): {prompt[:500]}...") # Print snippet of prompt

    # 4. Call the LLM
    print("FRA: Calling Gemini LLM...")
    try:
        response = call_gemini_llm(prompt)
        print(f"FRA: Received LLM response (first 500 chars): {response[:500]}...")
        response = response.strip()

        # Clean up markdown code blocks if necessary
        if response.startswith("```json"):
            response = response.split('\n', 1)[1]
            if response.endswith("```"):
                response = response.rsplit('\n', 1)[0]
        elif response.startswith("```"): # Handle generic code blocks too
             response = response.split('\n', 1)[1]
             if response.endswith("```"):
                 response = response.rsplit('\n', 1)[0]

    except Exception as e:
        print(f"FRA Error: LLM call failed: {e}")
        # Return an error outcome or raise exception
        return FRAOutput(
            reviewed_changes_outcome=[],
            summary=f"Error: LLM call failed during refinement: {e}"
        )


    # 5. Parse and validate the LLM response
    print("FRA: Parsing and validating LLM response...")
    try:
        data = json.loads(response)
        fra_output = FRAOutput(**data) # Validate with Pydantic
        print("FRA: LLM response validated successfully.")
        return fra_output
    except json.JSONDecodeError as e:
        print(f"FRA Error: Failed to parse JSON response: {e}")
        print(f"Raw response:\n{response}")
        # Handle error - maybe return a specific error object or raise
        raise ValueError(f"FRA failed to return valid JSON: {e}\nRaw response: {response}")
    except ValidationError as e:
        print(f"FRA Error: Response validation failed: {e.errors()}")
        print(f"Raw response:\n{response}")
        # Handle error
        raise ValueError(f"FRA response invalid format: {e}\nRaw response: {response}")

# --- Example Usage (Conceptual) ---
# This part would be in your main application flow, not in agents/fra.py
# from agents.fra import FRAInput, fra_agent, ManagerFeedbackItem

# # Assuming you have a run_id from a previous owe_agent call
# example_run_id = 1

# # Example manager feedback
# example_feedback = [
#     ManagerFeedbackItem(change_id="change_at_index_0", reason="Justification for title change is too generic, doesn't explain the significance of the revision date."),
#     ManagerFeedbackItem(change_id="change_at_index_3", reason="Justification for non-cash contribution lacks explicit Shariah source from the reasoning trace."),
#     ManagerFeedbackItem(change_id="change_at_index_99", reason="This change ID doesn't exist, check error handling.") # Example of bad ID
# ]

# # Create FRA input
# fra_input_data = FRAInput(run_id=example_run_id, feedback=example_feedback)

# # Call the FRA agent
# try:
#     refinement_results = fra_agent(fra_input_data)
#     print("\n--- Refinement Results ---")
#     print(refinement_results.model_dump_json(indent=2))

#     # You would then process refinement_results:
#     # - Retrieve the original run data again using example_run_id
#     # - Update the specific change objects in the fas_diff output based on the FRA's outcome ('updated_justification', 'updated_text')
#     # - If a change status is 'discarded', remove it from the changes list
#     # - If status is 'requires_manual_review', flag it.
#     # - Save the updated run data back to the DB or present the revised changes to the manager.

# except (ValueError, Exception) as e:
#     print(f"\nError during FRA execution: {e}")

