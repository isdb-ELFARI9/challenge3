import os
import json
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path to import agents and utils if necessary
# Adjust this based on your project structure
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mock Implementations for Dependencies ---

# Mock LLM call - This will return a predefined JSON response
def mock_call_gemini_llm(prompt: str) -> str:
    """Mocks the Gemini LLM call to return a predefined FRAOutput JSON."""
    print("\n--- Mock LLM Called ---")
    print(f"Prompt (first 500 chars):\n{prompt[:500]}...")
    print("--- End Mock LLM Called ---\n")

    # This is a mock response simulating what the FRA agent's LLM call should return
    # based on the expected FRAOutput schema.
    # We'll simulate updating the justification for the first feedback item
    # and discarding the second (if multiple feedbacks are provided).
    mock_response_data = {
      "reviewed_changes_outcome": [
        {
          "change_id": "change_at_index_0", # Matches the feedback item ID
          "status": "updated_justification",
          "revised_change_object": {
             "old_text": "# AAOIFI Financial Accounting Standard No. (4): Musharaka Financing - Simplified Overview",
             "new_text": "# FAS 4 (Revised 2024): Accounting for Investments in Diminishing Musharaka Structures within Real Estate Funds",
             "justification": "REVISED JUSTIFICATION based on trace: The title was updated to precisely reflect the standard's new focus on Diminishing Musharaka within real estate funds, specifically addressing the gaps identified by FCIA in this area. This aligns the standard's scope with the Shariah guidance (SPIA) and accounting rules (ARDA) developed for this specific context, ensuring clarity for users. (Source: SPIA Output, ARDA Rationale in Reasoning Trace)", # Example improved justification
             "section_id": "Title",
             "change_type": "modification"
          },
          "feedback_analysis": "Reviewed feedback on title justification. Found sufficient detail in the reasoning trace (SPIA/ARDA outputs) to provide a more specific justification linking the title change to the updated scope and the pipeline's analysis.",
          "justification_improvement_notes": "Justification rewritten to explicitly mention the link to identified gaps, Shariah guidance, and accounting rules from the trace.",
          "text_improvement_notes": None
        }
        # Add more outcomes here if you provide more feedback items in the test
        # Example for a second feedback item (e.g., change_at_index_1):
        # {
        #   "change_id": "change_at_index_1",
        #   "status": "discarded",
        #   "revised_change_object": None,
        #   "feedback_analysis": "Reviewed feedback on change_at_index_1. Found insufficient or contradictory reasoning in the trace to support this change. Recommending discard.",
        #   "justification_improvement_notes": None,
        #   "text_improvement_notes": None
        # }
      ],
      "summary": "Refinement process completed. Reviewed 1 change, updated justification for 1." # Update summary based on outcomes
    }

    return json.dumps(mock_response_data)

# Mock RAG functions - Return simple strings indicating they were called
def mock_retrieve_knowledge_from_pinecone_fas(query: str, fas_namespace: str) -> str:
    print(f"Mock FAS RAG called for namespace {fas_namespace} with query: {query[:50]}...")
    return "[Mock FAS Knowledge: Relevant info about FAS 4 and Diminishing Musharaka from RAG]"

def mock_retrieve_knowledge_from_pinecone_ss(query: str, ss_namespace: str) -> str:
    print(f"Mock SS RAG called for namespace {ss_namespace} with query: {query[:50]}...")
    return "[Mock SS Knowledge: Relevant info about Shirkah al-ʿAqd and related SS from RAG]"

# Mock embedding function
def mock_get_embedding(text: str) -> list:
     # Return a dummy embedding vector
     return [0.1] * 1536 # Assuming text-embedding-3-small default dimension


# --- SQLite Database Functions (Copied from utils/db.py for self-containment) ---
DATABASE_FILE = 'test_pipeline_runs.db' # Use a different DB file for testing

def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_prompt TEXT NOT NULL,
                fas_number TEXT NOT NULL,
                run_data TEXT NOT NULL, -- Store JSON dump of relevant data
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def save_pipeline_run(user_prompt: str, fas_number: str, run_data: dict) -> int:
    """Saves the data from a pipeline run to the database."""
    init_db() # Ensure DB and table exist
    run_data_json = json.dumps(run_data)
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO pipeline_runs (user_prompt, fas_number, run_data) VALUES (?, ?, ?)',
            (user_prompt, fas_number, run_data_json)
        )
        conn.commit()
        return cursor.lastrowid # Return the ID of the new row

def get_pipeline_run(run_id: int) -> dict | None:
    """Retrieves data for a specific pipeline run from the database."""
    init_db() # Ensure DB exists before trying to connect
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT user_prompt, fas_number, run_data FROM pipeline_runs WHERE run_id = ?', (run_id,))
        row = cursor.fetchone()
        if row:
            user_prompt, fas_number, run_data_json = row
            run_data = json.loads(run_data_json)
            # Reconstruct the data structure
            return {
                "run_id": run_id,
                "user_prompt": user_prompt,
                "fas_number": fas_number,
                **run_data # Merge the loaded JSON data
            }
        return None

# --- Mock get_fas_namespace (Copied from utils/fas_utils.py) ---
def get_fas_namespace(fas: str) -> str:
    # Map FAS number to namespace
    mapping = {
        '4': 'fas_4',
        '7': 'fas_7',
        '10': 'fas_10',
        '28': 'fas_28',
        '32': 'fas_32',
    }
    # Extract number from FAS string (e.g., 'FAS 4' -> '4')
    if isinstance(fas, str) and fas.lower().startswith('fas'):
        num = fas.split()[-1]
        return mapping.get(num, None)
    elif isinstance(fas, list):
        # If multiple FAS, return all relevant namespaces (PoC might only need the first)
        return [mapping.get(f.split()[-1], None) for f in fas if f.lower().startswith('fas')]
    return None


# --- Refinement Agent (FRA) Code (Copied from agents/fra.py for self-containment) ---
# This is the actual code you want to test, but using the mock dependencies
# Pydantic models are needed here
from pydantic import BaseModel, Field, ValidationError

# Schema for the specific feedback item provided by the manager
class ManagerFeedbackItem(BaseModel):
    change_id: str = Field(..., description="Identifier for the specific change being reviewed (e.g., index in changes list, or generated ID).")
    reason: str = Field(..., description="Detailed reason for the feedback (e.g., 'Justification weak', 'New text seems incorrect Shariah-wise').")

# Input schema for the Refinement Agent
class FRAInput(BaseModel):
    run_id: int = Field(..., description="The ID of the pipeline run data to retrieve from the database.")
    feedback: List[ManagerFeedbackItem] = Field(..., description="List of specific feedback items from the manager.")

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


# FRA Prompt Builder (using the mock dependencies implicitly via the agent function)
def build_fra_prompt(
    original_context: str,
    fas_number: str,
    original_fas_markdown: str,
    full_reasoning_trace: Dict[str, Any],
    fas_diff_output: Dict[str, Any],
    feedback: List[ManagerFeedbackItem],
    fas_knowledge: str,
    ss_knowledge: str
) -> str:
    """Builds the prompt for the Refinement Agent."""
    formatted_feedback = "\n".join([
        f"- Change ID: {item.change_id}\n  Reason: {item.reason}" for item in feedback
    ])

    changes_to_review = []
    all_changes_list = fas_diff_output.get("changes", [])
    for feedback_item in feedback:
        found_change = None
        try:
            index = int(feedback_item.change_id.split('_')[-1])
            if 0 <= index < len(all_changes_list):
                 found_change = all_changes_list[index]
            else:
                 print(f"Warning: Change ID index out of bounds: {feedback_item.change_id}")
        except (ValueError, IndexError):
            print(f"Warning: Could not parse index from Change ID: {feedback_item.change_id}")

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
                 "change_object": None,
                 "error": "Change ID not found or processed in fas_diff output"
             })

    formatted_changes_to_review = json.dumps(changes_to_review, indent=2)

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
    - Set the `status` ('updated_justification' | 'updated_text' | 'discarded' | 'requires_manual_review').
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

# --- Refinement Agent Function (using mocks) ---

def fra_agent(fra_input: FRAInput) -> FRAOutput:
    """
    Refinement Agent that reviews specific changes based on feedback and trace.
    Retrieves pipeline run data, performs RAG, prompts LLM, and validates output.
    Uses mock dependencies for testing.
    """
    print(f"FRA: Starting refinement for run ID {fra_input.run_id} with {len(fra_input.feedback)} feedback items.")

    # 1. Retrieve pipeline run data using the actual DB function
    run_data = get_pipeline_run(fra_input.run_id)
    if not run_data:
        print(f"FRA Error: Pipeline run with ID {fra_input.run_id} not found.")
        return FRAOutput(
            reviewed_changes_outcome=[],
            summary=f"Error: Pipeline run with ID {fra_input.run_id} not found."
        )

    original_context = run_data.get("user_prompt", "N/A")
    fas_number = run_data.get("fas_number", "N/A")
    full_reasoning_trace = run_data.get("reasoning_trace", {})
    original_fas_markdown = run_data.get("original_fas_markdown_text", "Original FAS text not stored or found.")
    fas_diff_output = full_reasoning_trace.get("fas_diff", {"changes": []})

    # 2. Perform RAG using mock RAG functions
    feedback_reasons = [item.reason for item in fra_input.feedback]
    rag_query = f"Context: {original_context}\nFeedback topics: {', '.join(feedback_reasons)}"
    print(f"FRA: RAG query: {rag_query[:100]}...")

    fas_namespace = None
    if isinstance(fas_number, str) and fas_number.lower().startswith('fas'):
         fas_namespace = get_fas_namespace(fas_number)
    elif isinstance(fas_number, list) and fas_number:
         fas_namespace = get_fas_namespace(fas_number[0])

    fas_knowledge = mock_retrieve_knowledge_from_pinecone_fas(rag_query, fas_namespace)
    print(f"FRA: Retrieved FAS knowledge (first 100 chars): {fas_knowledge[:100]}...")

    ss_knowledge = mock_retrieve_knowledge_from_pinecone_ss(rag_query, "SS12") # Using a default namespace for mock
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

    # 4. Call the Mock LLM
    try:
        response = mock_call_gemini_llm(prompt) # Use the mock LLM
        response = response.strip()

        # Clean up markdown code blocks if necessary (mock might not add them, but good practice)
        if response.startswith("```json"):
            response = response.split('\n', 1)[1]
            if response.endswith("```"):
                response = response.rsplit('\n', 1)[0]
        elif response.startswith("```"):
             response = response.split('\n', 1)[1]
             if response.endswith("```"):
                 response = response.rsplit('\n', 1)[0]

    except Exception as e:
        print(f"FRA Error: Mock LLM call failed: {e}")
        return FRAOutput(
            reviewed_changes_outcome=[],
            summary=f"Error: Mock LLM call failed during refinement: {e}"
        )

    # 5. Parse and validate the LLM response
    print("FRA: Parsing and validating Mock LLM response...")
    try:
        data = json.loads(response)
        fra_output = FRAOutput(**data) # Validate with Pydantic
        print("FRA: Mock LLM response validated successfully.")
        return fra_output
    except json.JSONDecodeError as e:
        print(f"FRA Error: Failed to parse JSON response: {e}")
        print(f"Raw response:\n{response}")
        raise ValueError(f"FRA failed to return valid JSON: {e}\nRaw response: {response}")
    except ValidationError as e:
        print(f"FRA Error: Response validation failed: {e.errors()}")
        print(f"Raw response:\n{response}")
        raise ValueError(f"FRA response invalid format: {e}\nRaw response: {response}")


# --- Test Execution ---

if __name__ == "__main__":
    # Ensure the test database is clean
    if os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)
        print(f"Cleaned up existing test database: {DATABASE_FILE}")

    init_db() # Initialize the test database

    # 1. Create Mock Pipeline Run Data
    mock_user_prompt = "Saudi Arabia legalized crypto, update FAS 4 for crypto accounting."
    mock_fas_number = "FAS 4"
    mock_original_fas_markdown = "# Original FAS 4 Content...\n## Title\n# AAOIFI Financial Accounting Standard No. (4): Musharaka Financing - Simplified Overview\n..." # Simulate original text

    # Simulate a reasoning trace including a simplified fas_diff output
    mock_reasoning_trace = {
        "uiria": {"context": mock_user_prompt, "identified_FAS": mock_fas_number},
        "fcia": {"identified_gaps": [{"clause": "FAS 4 - Title", "issue": "Title is too general for specific updates.", "justification": "Need a title reflecting the specific Diminishing Musharaka update."}]},
        "spia": {"shariah_solution": "Guidance on Diminishing Musharaka in RE funds.", "updated_shariah_clauses": []},
        "arda": {"rationale": "Accounting rules needed for specific RE fund structures.", "updated_accounting_clauses": []},
        "stsa": {"change_log": ["Updated title."], "all_updated_sections": {}, "original_sections": {}},
        "document": "# Updated Document Content...",
        "change_summary": "Summary of changes...",
        "fas_diff": {
            "changes": [
                { # This is the change example provided by the user
                    "old_text": "# AAOIFI Financial Accounting Standard No. (4): Musharaka Financing - Simplified Overview",
                    "new_text": "# FAS 4 (Revised 2024): Accounting for Investments in Diminishing Musharaka Structures within Real Estate Funds",
                    "justification": "Modified the title to reflect the updated scope and focus on Diminishing Musharaka structures within real estate funds. This change aligns the standard with the specific issues addressed in the update, focusing on the identified gaps in the original FAS 4 and ensuring compliance with Shariah principles, particularly those related to Shirkah al-ʿAqd.",
                    "section_id": "Title",
                    "change_type": "modification"
                },
                # Add other mock changes here if needed for testing multiple feedback items
                {
                    "old_text": "Some original text about investments.",
                    "new_text": "Updated text about investments including new guidance.",
                    "justification": "Added guidance based on ARDA output.",
                    "section_id": "Investments",
                    "change_type": "modification"
                }
            ],
            "key_changes_summary": "Key changes included title and investment section.",
            "change_statistics": {"additions": 0, "deletions": 0, "modifications": 2}
        },
         "original_fas_markdown_text": mock_original_fas_markdown # Include original markdown
    }

    # 2. Save Mock Data to DB
    run_id = save_pipeline_run(mock_user_prompt, mock_fas_number, mock_reasoning_trace)
    print(f"\nMock pipeline run saved with ID: {run_id}")

    # 3. Create Mock Manager Feedback
    # Targeting the first change (index 0)
    mock_feedback = [
        ManagerFeedbackItem(
            change_id="change_at_index_0", # Use the index as the ID for this PoC
            reason="Justification for the title change is too generic and doesn't clearly link to the specific reasoning in the trace."
        ),
         # Example of feedback for the second change (index 1)
        # ManagerFeedbackItem(
        #     change_id="change_at_index_1",
        #     reason="Justification for investment text update is unclear and lacks source."
        # )
    ]

    # 4. Create FRA Input
    fra_input = FRAInput(run_id=run_id, feedback=mock_feedback)

    # 5. Call the FRA Agent (which uses the mocks internally)
    print("\nCalling FRA agent...")
    try:
        refinement_output = fra_agent(fra_input)

        # 6. Print Results
        print("\n--- FRA Agent Output ---")
        print(refinement_output.model_dump_json(indent=2))
        print("--- End FRA Agent Output ---")

    except (ValueError, Exception) as e:
        print(f"\nError during FRA execution: {e}")

    finally:
        # Clean up the test database file
        if os.path.exists(DATABASE_FILE):
            # os.remove(DATABASE_FILE)
            # print(f"Cleaned up test database: {DATABASE_FILE}")
            # Keep the DB file for inspection after the run if needed
            print(f"\nTest database kept for inspection: {DATABASE_FILE}")

