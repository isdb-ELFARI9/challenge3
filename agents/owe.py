from agents.uiria import uiria_agent
from agents.fcia import fcia_agent
from agents.spia import spia_agent
from agents.arda import arda_agent
from agents.stsa import stsa_agent
from schemas.uiria import UIRIAInput, UIRIAOutput
from schemas.fcia import FCIAInput, FCIAOutput
from schemas.spia import SPIAInput, SPIAOutput
from schemas.arda import ARDAInput, ARDAOutput
from schemas.stsa import STSAInput, STSAOutput

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

    # Compose the final output for UIRIA/user
    return {
        "updated_fas_document": stsa_out.all_updated_sections,
        "change_log": stsa_out.change_log,
        "references": stsa_out.references,
        "reasoning_trace": reasoning_trace,
        "explainability": "Each agent received augmented context including summaries and reasoning trace from previous steps for maximum accuracy."
    } 