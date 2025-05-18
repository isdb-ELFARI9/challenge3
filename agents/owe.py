import os
from typing import Dict, Any
from agents.board import board_agent
from agents.supervisor import supervisor_agent_llm
from fas_gaps_similarities_identifier_agent.run_workflow import run_complete_workflow
from schemas.owe import OWEInput, OWEOutput
from agents.uiria import uiria_agent
from agents.fcia import fcia_agent
from agents.spia import spia_agent
from agents.arda import arda_agent
from agents.stsa import stsa_agent
from agents.document_composer import document_composer_agent
from agents.change_summary import change_summary_agent
from agents.fas_diff import fas_diff_agent, update_markdown_with_changes, save_changes_to_markdown
from schemas.uiria import UIRIAInput, UIRIAOutput
from schemas.fcia import FCIAInput, FCIAOutput
from schemas.spia import SPIAInput, SPIAOutput
from schemas.arda import ARDAInput, ARDAOutput
from schemas.stsa import STSAInput, STSAOutput
from schemas.fas_diff import FASDiffInput
from utils.fas_utils import get_fas_namespace
import json
from utils.db import save_pipeline_run
from utils.write_to_file import write_to_file

# Utility to map FAS to namespace

def get_fas_namespace(fas):
    mapping = {
        '4': 'fas_4',
        '7': 'fas_7',
        '8': 'fas_8',
        '10': 'fas_10',
        '16': 'fas_16',
        '28': 'fas_28',
        '32': 'fas_32',
    }
    if isinstance(fas, str) and fas.lower().startswith('fas'):
        num = fas.split()[-1]
        return mapping.get(num, None)
    return None

# The OWE agent orchestrates the flow between all agents with contextual augmentation

async def owe_agent(user_prompt: str) -> dict:
    reasoning_trace = {}
    # Step 1: Intake and structure user input
     # Create output directory
    output_dir = "./outputs"
     # Configure the FAS standards to analyze
    fas_standards = ["fas_4", "fas_8", "fas_16", "fas_7", "fas_10","fas_28", "fas_32"]
    
    # Set the LLM provider
    selected_provider = "openai"  # Change to "gemini" to use Google's model
    os.makedirs(output_dir, exist_ok=True)
    write_to_file("prompts.txt", "user prompt \n")

    write_to_file("prompts.txt", user_prompt)
    fas_gaps = await run_complete_workflow(user_prompt,fas_standards,selected_provider,output_dir)
    print("fas_gaps", fas_gaps["synthesis_result"])
    # uiria_out: UIRIAOutput = uiria_agent(user_prompt)
    reasoning_trace["fas_gaps"] = fas_gaps["synthesis_result"]

    # # Step 2: FCIA - Contextualize and find gaps
    # fcia_context = (
    #     f"User context: {uiria_out.context}\n"
    #     f"Reasoning trace so far: {reasoning_trace['uiria']}"
    # )
    # fcia_in = FCIAInput(
    #     context=fcia_context,
    #     FAS=uiria_out.identified_FAS,
    #     knowledge_indexes=[get_fas_namespace(uiria_out.identified_FAS)]
    # )
    # fcia_out = fcia_agent(fcia_in)
    # reasoning_trace["fcia"] = fcia_out.model_dump()

    # # Step 3: SPIA - Shariah solution for the gap
    # spia_user_context = (
    #     f"User context: {uiria_out.context}\n"
    #     f"FCIA gap summary: {fcia_out.identified_gaps}\n"
    #     f"Reasoning trace so far: {reasoning_trace['fcia']}"
    # )
    # spia_in = SPIAInput(
    #     gap_report=fcia_out.identified_gaps,
    #     FAS=uiria_out.identified_FAS,
    #     user_context=spia_user_context,
    #     knowledge_indexes=["SS12"]
    # )
    # spia_out: SPIAOutput = spia_agent(spia_in)
    # reasoning_trace["spia"] = spia_out.model_dump()

    # # Step 4: ARDA - Update accounting rules
    # arda_user_context = (
    #     f"User context: {uiria_out.context}\n"
    #     f"FCIA gap summary: {fcia_out.identified_gaps}\n"
    #     f"SPIA shariah solution: {spia_out.shariah_solution}\n"
    #     f"Reasoning trace so far: {reasoning_trace['spia']}"
    # )
    # arda_in = ARDAInput(
    #     shariah_update=spia_out.model_dump(),
    #     FAS=uiria_out.identified_FAS,
    #     user_context=arda_user_context,
    #     knowledge_indexes=[get_fas_namespace(uiria_out.identified_FAS)]
    # )
    # arda_out: ARDAOutput = arda_agent(arda_in)
    # reasoning_trace["arda"] = arda_out.model_dump()
    # Prepare context and gap report for board agents
    user_context = user_prompt
    gap_report = fas_gaps["synthesis_result"]
    fas_number = fas_gaps["synthesis_result"]["overall_verdict"]["fas_to_update"][0]#todo change to the fas numbers list
    #remove the _ and replace it with a white space 
    fas_number = fas_number.replace("_", " ")

    

    
    print("fas_number", fas_number)
    knowledge_fas_ns = get_fas_namespace(fas_number)

    # Step 3: Board proposals (instead of direct SPIA and ARDA calls)
    board_proposals = [
    {"llm": "gemini", "content": board_agent(fas=fas_number,
            user_context=user_context,
            gap_report=json.dumps(gap_report),
            knowledge_indexes=["SS12", knowledge_fas_ns], llm_name="gemini")},
    {"llm": "gpt", "content": board_agent(fas=fas_number,
            user_context=user_context,
            gap_report=json.dumps(gap_report),
            knowledge_indexes=["SS12", knowledge_fas_ns], llm_name="gpt")},
    {"llm": "deepseek", "content": board_agent(fas=fas_number,
            user_context=user_context,
            gap_report=json.dumps(gap_report),
            knowledge_indexes=["SS12", knowledge_fas_ns], llm_name="deepseek")},
    ]
    reasoning_trace["board_proposals"] = board_proposals

    # Step 4: Supervisor decision on board proposals
    supervisor_decision = supervisor_agent_llm(board_proposals)

    # Use supervisor_decision for downstream steps instead of spia_out and arda_out
    # For example, split supervisor_decision into the equivalent of spia_out and arda_out data:
    #turn the supervisor descision to json knowing that it starts like ```json ``` and somtime directly json
    # supervisor_decision is a python object because o did json.loads() on it
    # extract the merged_shariah_clauses from the supervisor_decision
    print("supervisor_decision", supervisor_decision,"type", type(supervisor_decision))
    supervisor_decision_str = json.dumps(supervisor_decision, indent=2)
    # spia_out = supervisor_decision['merged_shariah_clauses']
    # arda_out = supervisor_decision['merged_shariah_clauses']

    # # Step 5: STSA - Update FAS text and structure
    # stsa_user_context = (
    #     f"User context: {uiria_out.context}\n"
    #     f"FCIA gap summary: {fcia_out.identified_gaps}\n"
    #     f"the updated shariah clauses and accounting clauses: {supervisor_decision_str}\n"
    #     f"Reasoning trace so far: {reasoning_trace['supervisor_decision']}"
    # )
    # stsa_in = STSAInput(
    #     FAS=uiria_out.identified_FAS,
    #     user_context=stsa_user_context,
    #     knowledge_indexes=[get_fas_namespace(uiria_out.identified_FAS)]
    # )
    # stsa_out: STSAOutput = stsa_agent(stsa_in)
    # reasoning_trace["stsa"] = stsa_out.model_dump()

   

    # Step 6: Change Summary - Generate user-facing summary
    change_summary = change_summary_agent(
        change_log=supervisor_decision_str,
        reasoning_trace=reasoning_trace,
        user_context=user_prompt
    )

    # Add document and change summary to the reasoning trace
    reasoning_trace["change_summary"] = change_summary

    print( reasoning_trace, change_summary)

    # Get original sections from STSA output for comparison with document content


    # After document composition, analyze differences
    fas_diff_input = FASDiffInput(
        # old_fas_markdown=json.dumps(original_sections, indent=2),
        new_fas_markdown=supervisor_decision_str,#the changes not really the ew markdown file
        fas_number=fas_number,
        context=user_prompt,
        multi_agent_reasoning=reasoning_trace
    )
    
    diff_analysis = fas_diff_agent(fas_diff_input)
    print("diff_analysis", diff_analysis)
    
    # Process diff_analysis for JSON serialization
    diff_dict = {}
    if hasattr(diff_analysis, "to_dict"):
        diff_dict = diff_analysis.to_dict()
    else:
        diff_dict = {
            "changes": [change.to_dict() for change in diff_analysis.changes],
            "key_changes_summary": diff_analysis.key_changes_summary,
            "change_statistics": diff_analysis.change_statistics
        }
    
    reasoning_trace["fas_diff"] = diff_dict
    
    # Update the document with key changes section
    # final_document = update_markdown_with_changes(json.dumps(document, indent=2), diff_analysis.key_changes_summary)
    
    # Save comprehensive changes to file
    changes_file = save_changes_to_markdown(
        diff_analysis.changes,
        fas_number,
        diff_analysis,
        change_summary,
        reasoning_trace
    )
    print(f"Detailed changes saved to: {changes_file}")

    data_to_save = {
        "user_prompt": user_prompt,
        "fas_number": fas_number,
        "reasoning_trace": reasoning_trace, # This now includes all outputs including fas_diff
        "change_file": changes_file # Save the final proposed markdown
    }
    run_id = save_pipeline_run(user_prompt, fas_number, data_to_save)
    print(f"Pipeline run saved with ID: {run_id}")

    return {
        "user_prompt": user_prompt,
        "fas_number": fas_number,
        "reasoning_trace": reasoning_trace, # This now includes all outputs including fas_diff
        "change_file": changes_file # Save the final proposed markdown
    } 