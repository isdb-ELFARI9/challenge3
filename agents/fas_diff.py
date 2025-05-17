import os
import json
import re
from typing import List, Dict, Any
from schemas.fas_diff import FASDiffInput, FASDiffOutput, ChangeRecord
from utils.llm import call_gemini_llm, get_llm_response

def read_original_fas_file(fas_number: str) -> str:
    """
    Read the original FAS markdown file.
    
    Args:
        fas_number: The FAS number to read
        
    Returns:
        str: Content of the original FAS markdown file
    """
    # Clean the FAS number (remove "FAS" prefix if present)
    if isinstance(fas_number, str) and fas_number.lower().startswith('fas'):
        fas_number = fas_number.split()[-1]
    
    file_path = os.path.join("fas_markdowns","old", f"fas_{fas_number}.md")
    
    # If the file doesn't exist, create the directory and return empty string
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Warning: Original FAS file not found at {file_path}")
        return "No original FAS document available."
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_changes_to_markdown(changes: List[ChangeRecord], fas_number: str, document: Any, change_summary: Any, reasoning_trace: Dict[str, Any]) -> str:
    """
    Save changes to a detailed changelog file in the fas_markdowns/changes folder.
    
    Args:
        changes: List of ChangeRecord objects
        fas_number: The FAS number being analyzed
        document: The complete updated document (string or dict)
        change_summary: User-friendly summary of changes (string or dict)
        reasoning_trace: Complete reasoning trace
        
    Returns:
        str: Path to the saved file
    """
    # Create changes directory if it doesn't exist
    changes_dir = os.path.join("fas_markdowns", "changes")
    os.makedirs(changes_dir, exist_ok=True)
    
    # Convert changes to the desired format
    changes_dict = {
        "fas_number": fas_number,
        "changes": [
            {
                "old_paragraph": change.old_text,
                "new_paragraph": change.new_text,
                "justification": change.justification,
                "section": change.section_id,
                "type": change.change_type
            }
            for change in changes
        ],
        "document": document,
        "change_summary": change_summary,
        "reasoning_trace": reasoning_trace
    }
    
    # Save to file
    output_file = os.path.join(changes_dir, f"fas_{fas_number}_changes.json")
    
    # Convert reasoning_trace to be JSON serializable
    clean_reasoning_trace = {}
    for key, value in reasoning_trace.items():
        if key == "fas_diff" and "changes" in value:
            # Handle ChangeRecord objects in fas_diff
            clean_changes = []
            for change in value.get("changes", []):
                if hasattr(change, "to_dict"):
                    clean_changes.append(change.to_dict())
                else:
                    clean_changes.append(change)
            
            clean_value = value.copy()
            clean_value["changes"] = clean_changes
            clean_reasoning_trace[key] = clean_value
        else:
            clean_reasoning_trace[key] = value
    
    # Create clean data for JSON serialization
    clean_data = {
        "fas_number": fas_number,
        "changes": [change.to_dict() for change in changes],
        "document": document,
        "change_summary": change_summary,
        "reasoning_trace": clean_reasoning_trace
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        try:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
        except TypeError as e:
            print(f"Warning: Error serializing to JSON: {e}")
            # Fallback to string conversion for non-serializable objects
            try:
                # Convert everything to strings as a last resort
                json.dump(str(clean_data), f, indent=2, ensure_ascii=False)
            except:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Error serializing changes for FAS {fas_number}")
    
    return output_file

def extract_json_from_code_block(raw_response: str) -> str:
    # Remove triple backticks and optional "json" language tag
    cleaned = re.sub(r"^```json\s*|```$", "", raw_response.strip(), flags=re.MULTILINE)
    return cleaned.strip()

def fas_diff_agent(input_data: FASDiffInput) -> FASDiffOutput:
    """
    Formats changes between old and new FAS versions into structured diffs.
    
    This agent:
    1. Gets the original FAS file content
    2. Identifies sentences/paragraphs that need to be changed based on proposed updates
    3. Creates structured change records with justifications from reasoning trace
    4. Generates a formatted diff output
    """
    # Read the original FAS markdown file
    original_fas_file = read_original_fas_file(input_data.fas_number)
    
    # DEBUGGING OUTPUT
    print("\n======== DEBUG INFORMATION FOR FAS_DIFF_AGENT ========")
    print(f"FAS Number: {input_data.fas_number}")
    print(f"Context: {input_data.context[:100]}...")
    
    # Get proposed updates and change summary
    proposed_updates = input_data.new_fas_markdown
    change_summary = input_data.multi_agent_reasoning.get("change_summary", "No summary available")
    
    # Get the full reasoning trace for justifications
    reasoning_trace = input_data.multi_agent_reasoning

    stsa_changes = reasoning_trace.get("stsa", {}).get("change_log", ["No changes recorded"])
    fcia_gaps = reasoning_trace.get("fcia", {}).get("identified_gaps", "No gaps identified")
    spia_solution = reasoning_trace.get("spia", {}).get("shariah_solution", "No Shariah solution provided")
    arda_rationale = reasoning_trace.get("arda", {}).get("rationale", "No accounting rationale provided")
    reasoning_summary = f"""
    IDENTIFIED GAPS: 
    {fcia_gaps}
    
    SHARIAH SOLUTION:
    {spia_solution}
    
    ACCOUNTING RATIONALE:
    {arda_rationale}
    
    PROPOSED CHANGES:
    {json.dumps(stsa_changes, indent=2) if isinstance(stsa_changes, (list, dict)) else stsa_changes}
    """
    
    # Prepare prompt for the LLM to identify specific text changes
    prompt = f"""You are the final agent in a system that analyzes and updates Financial Accounting Standards (FAS). Your primary role is to precisely identify textual changes based on the detailed proposals and reasoning from previous agents and you can propose changes that are not mentioned in the reasoning trace , but i want the old fas document to inlude all changes correctly in all of its sections eacg with detailed justification.

        Previous agents have analyzed FAS {input_data.fas_number} and proposed the following updates:
        {proposed_updates}

        Change summary:
        {change_summary}

        Reasoning trace (contains detailed justifications, Shariah basis, accounting logic, and references for the updates):
        {reasoning_summary}

        The original FAS document is:
        {original_fas_file}

        Your task is to:
        1. Based on the difference between the "original_fas_file" and the implied *updated* content from "proposed_updates" and "change_summary", identify the EXACT sentences, clauses, or paragraphs in the "original_fas_file" that need to change AND SCAN ALL THE OLD FILE AND UPDATE EVERYTHING THAT NEEDS TO BE UPDATED.
        2. Provide the EXACT new text that should replace the "old_text" or be inserted (for additions).
        3. For *each individual change* ("old_text" to "new_text"), you MUST extract and detail the specific justification from the "reasoning_summary". **Formulate a concise, clear justification for *this specific text modification/addition/deletion*, referencing the Shariah basis, accounting logic, gap addressed, and potential context implications as found in the "reasoning_summary". If the reasoning mentions specific sources for *this particular point*, include them in the justification text.** Do NOT just copy the whole rationale; extract the *reason for this specific line change* or add it , the most important thing is that the reasonning of the justification is detailed with sources and logical and really relevant to the change.
        4. Categorize each change accurately as "addition", "deletion", or "modification".
        5. Identify which section or clause the change belongs to ("section_id").

        You MUST respond with a JSON object in this exact format:
        {{
            "changes": [
                {{
                    "old_text": "The exact original text that was changed or the anchor text if an addition is made after it (e.g., 'Insert after this sentence:'). For deletions, this is the text removed.",
                    "new_text": "The exact new text that replaced 'old_text'. For deletions, this is an empty string ''. For additions, this is the text added.",
                    "justification": "detailed justification for *this specific text change* derived from the reasoning_summary or add it if it's not in the reasoning_summary and with detailed sources if exists. Example: 'Modified text to explicitly include digital assets as 'Mal mutaqawwim' based on SPIA's guidance derived from AAOIFI SS 21, addressing the foundational gap in asset definition for potential crypto activities mentioned in the reasoning.'",
                    "section_id": "Section identifier (e.g., 'Section 3.1 Definitions', 'Clause 4.5')",
                    "change_type": "addition/deletion/modification"
                }},
                // Additional changes
            ],
            "key_changes_summary": "A markdown-formatted summary of the key areas updated and why, drawing from the justifications above.",
            "change_statistics": {{
                "additions": 0,
                "deletions": 0,
                "modifications": 0
            }}
        }}

        Focus ONLY on extracting or adding the changes, precisely formatting the changes, and *enriching* the justification for *each specific text change* using the provided reasoning trace. The decisions about what to change have already been made by previous agents.
        """

    print("prompt for fas diff agent", prompt)
    
    # Get LLM analysis
    analysis_raw = call_gemini_llm(prompt)

    print("analysis for fas diff agent", analysis_raw)
    try:
        cleaned_analysis = extract_json_from_code_block(analysis_raw)
        analysis = json.loads(cleaned_analysis)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse LLM response into JSON.")
        print("Raw response:", analysis_raw)
        raise e
    
    # Process the analysis into structured format
    changes = []
    if isinstance(analysis, dict) and "changes" in analysis:
        for change in analysis["changes"]:
            if all(k in change for k in ["old_text", "new_text", "justification", "section_id", "change_type"]):
                changes.append(ChangeRecord(
                    old_text=change["old_text"],
                    new_text=change["new_text"],
                    justification=change["justification"],
                    section_id=change["section_id"],
                    change_type=change["change_type"]
                ))
    
    # Extract change statistics
    additions = sum(1 for change in changes if change.change_type == "addition")
    deletions = sum(1 for change in changes if change.change_type == "deletion")
    modifications = sum(1 for change in changes if change.change_type == "modification")
    
    # Extract or create key changes summary
    key_changes_summary = analysis.get("key_changes_summary", "")
    # if not key_changes_summary:
    #     key_changes_summary = "## Key Changes\n\n"
    #     key_changes_summary += change_summary
    
    # # Save changes to the changes directory
    # changes_file = save_changes_to_markdown(
    #     changes, 
    #     input_data.fas_number,
    #     proposed_updates,
    #     change_summary,
    #     input_data.multi_agent_reasoning
    # )
    # print(f"Changes saved to: {changes_file}")
    
    result = FASDiffOutput(
        changes=changes,
        key_changes_summary=key_changes_summary,
        change_statistics={
            "additions": additions,
            "deletions": deletions,
            "modifications": modifications
        }
    )
    
    return result

def update_markdown_with_changes(markdown_content: str, key_changes: str) -> str:
    """
    Adds a 'Key Changes' section to the markdown content.
    """
    # Find the first heading in the markdown
    lines = markdown_content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if line.startswith('#'):
            insert_index = i + 1
            break
    
    # Insert the key changes section
    key_changes_section = f"\n## Key Changes from Previous FAS Version\n\n{key_changes}\n"
    lines.insert(insert_index, key_changes_section)
    
    return '\n'.join(lines) 