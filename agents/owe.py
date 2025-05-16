from typing import Dict, Any
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

# Utility to map FAS to namespace

def get_fas_namespace(fas):
    mapping = {
        '4': 'fas_4',
        '7': 'fas_7',
        '10': 'fas_10',
        '28': 'fas_28',
        '32': 'fas_32',
    }
    if isinstance(fas, str) and fas.lower().startswith('fas'):
        num = fas.split()[-1]
        return mapping.get(num, None)
    return None

# The OWE agent orchestrates the flow between all agents with contextual augmentation

def owe_agent(user_prompt: str) -> dict:
    reasoning_trace = {}
    # Step 1: Intake and structure user input
    uiria_out: UIRIAOutput = uiria_agent(user_prompt)
    reasoning_trace["uiria"] = uiria_out.model_dump()

    # Step 2: FCIA - Contextualize and find gaps
    fcia_context = (
        f"User context: {uiria_out.context}\n"
        f"Reasoning trace so far: {reasoning_trace['uiria']}"
    )
    fcia_in = FCIAInput(
        context=fcia_context,
        FAS=uiria_out.identified_FAS,
        knowledge_indexes=[get_fas_namespace(uiria_out.identified_FAS)]
    )
    fcia_out: FCIAOutput = fcia_agent(fcia_in)
    reasoning_trace["fcia"] = fcia_out.model_dump()

    # Step 3: SPIA - Shariah solution for the gap
    spia_user_context = (
        f"User context: {uiria_out.context}\n"
        f"FCIA gap summary: {fcia_out.identified_gaps}\n"
        f"Reasoning trace so far: {reasoning_trace['fcia']}"
    )
    spia_in = SPIAInput(
        gap_report=fcia_out.identified_gaps,
        FAS=uiria_out.identified_FAS,
        user_context=spia_user_context,
        knowledge_indexes=["SS12"]
    )
    spia_out: SPIAOutput = spia_agent(spia_in)
    reasoning_trace["spia"] = spia_out.model_dump()

    # Step 4: ARDA - Update accounting rules
    arda_user_context = (
        f"User context: {uiria_out.context}\n"
        f"FCIA gap summary: {fcia_out.identified_gaps}\n"
        f"SPIA shariah solution: {spia_out.shariah_solution}\n"
        f"Reasoning trace so far: {reasoning_trace['spia']}"
    )
    arda_in = ARDAInput(
        shariah_update=spia_out.model_dump(),
        FAS=uiria_out.identified_FAS,
        user_context=arda_user_context,
        knowledge_indexes=[get_fas_namespace(uiria_out.identified_FAS)]
    )
    arda_out: ARDAOutput = arda_agent(arda_in)
    reasoning_trace["arda"] = arda_out.model_dump()

    # Step 5: STSA - Update FAS text and structure
    stsa_user_context = (
        f"User context: {uiria_out.context}\n"
        f"FCIA gap summary: {fcia_out.identified_gaps}\n"
        f"SPIA shariah solution: {spia_out.shariah_solution}\n"
        f"ARDA accounting rationale: {arda_out.rationale}\n"
        f"Reasoning trace so far: {reasoning_trace['arda']}"
    )
    stsa_in = STSAInput(
        updated_shariah_section=spia_out.model_dump(),
        updated_accounting_section=arda_out.model_dump(),
        FAS=uiria_out.identified_FAS,
        user_context=stsa_user_context,
        knowledge_indexes=[get_fas_namespace(uiria_out.identified_FAS)]
    )
    stsa_out: STSAOutput = stsa_agent(stsa_in)
    reasoning_trace["stsa"] = stsa_out.model_dump()

    # Step 6: Document Composer - Build the final document
    document = document_composer_agent(
        user_context=uiria_out.context,
        spia_out=spia_out.model_dump(),
        arda_out=arda_out.model_dump(),
        stsa_out=stsa_out.model_dump(),
        reasoning_trace=reasoning_trace
    )

    # Step 7: Change Summary - Generate user-facing summary
    change_summary = change_summary_agent(
        change_log=stsa_out.change_log,
        reasoning_trace=reasoning_trace,
        user_context=uiria_out.context
    )

    # Add document and change summary to the reasoning trace
    reasoning_trace["document"] = document
    reasoning_trace["change_summary"] = change_summary

    print(document, reasoning_trace, change_summary)

    # Get original sections from STSA output for comparison with document content
    original_sections = stsa_out.original_sections

    # After document composition, analyze differences
    fas_diff_input = FASDiffInput(
        old_fas_markdown=json.dumps(original_sections, indent=2),
        new_fas_markdown=json.dumps(document, indent=2),#the changes not really the ew markdown file
        fas_number=uiria_out.identified_FAS,
        context=uiria_out.context,
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
    final_document = update_markdown_with_changes(json.dumps(document, indent=2), diff_analysis.key_changes_summary)
    
    # Save comprehensive changes to file
    changes_file = save_changes_to_markdown(
        diff_analysis.changes,
        uiria_out.identified_FAS,
        final_document,
        change_summary,
        reasoning_trace
    )
    print(f"Detailed changes saved to: {changes_file}")
    
    # Compose the final output
    return {
        "document": final_document,
        "change_summary": change_summary,
        "reasoning_trace": reasoning_trace,
        "old_outputs": {
            "updated_fas_document": stsa_out.all_updated_sections,
            "change_log": stsa_out.change_log,
            "references": stsa_out.references,
            "detailed_changes": [change.to_dict() for change in diff_analysis.changes],
            "change_statistics": diff_analysis.change_statistics
        },
        "diff":diff_analysis
    
    } 