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

# The OWE agent orchestrates the flow between all agents

def owe_agent(user_prompt: str) -> dict:
    # Step 1: Intake and structure user input
    uiria_out: UIRIAOutput = uiria_agent(user_prompt)

    # Step 2: FCIA - Contextualize and find gaps
    fcia_in = FCIAInput(
        context=uiria_out.context,
        FAS=uiria_out.identified_FAS,
        knowledge_indexes=[ns for ns in uiria_out.identified_FAS]  # Assume FAS numbers map to namespaces
    )
    fcia_out: FCIAOutput = fcia_agent(fcia_in)

    # Step 3: SPIA - Shariah solution for the gap
    spia_in = SPIAInput(
        gap_report=fcia_out.identified_gaps,
        FAS=uiria_out.identified_FAS,
        user_context=uiria_out.context,
        knowledge_indexes=["SS12"]  # For now, use SS12 as the most relevant Shariah namespace
    )
    spia_out: SPIAOutput = spia_agent(spia_in)

    # Step 4: ARDA - Update accounting rules
    arda_in = ARDAInput(
        shariah_update=spia_out.model_dump(),
        FAS=uiria_out.identified_FAS,
        user_context=uiria_out.context,
        knowledge_indexes=[ns for ns in uiria_out.identified_FAS]
    )
    arda_out: ARDAOutput = arda_agent(arda_in)

    # Step 5: STSA - Update FAS text and structure
    stsa_in = STSAInput(
        updated_shariah_section=spia_out.model_dump(),
        updated_accounting_section=arda_out.model_dump(),
        FAS=uiria_out.identified_FAS,
        user_context=uiria_out.context,
        knowledge_indexes=[ns for ns in uiria_out.identified_FAS]
    )
    stsa_out: STSAOutput = stsa_agent(stsa_in)

    # Compose the final output for UIRIA/user
    return {
        "updated_fas_document": stsa_out.all_updated_sections,
        "change_log": stsa_out.change_log,
        "references": stsa_out.references,
        "reasoning_trace": {
            "uiria": uiria_out.model_dump(),
            "fcia": fcia_out.model_dump(),
            "spia": spia_out.model_dump(),
            "arda": arda_out.model_dump(),
            "stsa": stsa_out.model_dump(),
        }
    } 