from agents.spia import spia_agent
from agents.arda import arda_agent
from schemas.spia import SPIAInput
from schemas.arda import ARDAInput

def board_agent(fas, user_context, gap_report, knowledge_indexes, llm_name="gemini"):
    """
    Each 'board member' (agent) performs both SPIA and ARDA analysis and returns a proposal.
    """
    # Step 1: Shariah Proposal
    spia_input = SPIAInput(
        FAS=fas,
        user_context=user_context,
        gap_report=gap_report,
        knowledge_indexes=knowledge_indexes
    )
    spia_result = spia_agent(spia_input, llm_name)

    # Step 2: Accounting Proposal (based on Shariah output)
    arda_input = ARDAInput(
        FAS=fas,
        user_context="The user needs accounting rules for " + user_context.lower(),
        shariah_update={
            "shariah_solution": spia_result.shariah_solution,
            "updated_shariah_clauses": spia_result.updated_shariah_clauses,
            "references": spia_result.references
        },
        knowledge_indexes=knowledge_indexes
    )
    arda_result = arda_agent(arda_input, llm_name)

    print("from one board member that use the llm :", llm_name,{
        "shariah_solution": spia_result.shariah_solution,
        "updated_shariah_clauses": spia_result.updated_shariah_clauses,
        "accounting_rationale": arda_result.rationale,
        "updated_accounting_clauses": arda_result.updated_accounting_clauses,
        "references": list(set(spia_result.references + arda_result.references)),
    })

    return {
        "shariah_solution": spia_result.shariah_solution,
        "updated_shariah_clauses": spia_result.updated_shariah_clauses,
        "accounting_rationale": arda_result.rationale,
        "updated_accounting_clauses": arda_result.updated_accounting_clauses,
        "references": list(set(spia_result.references + arda_result.references)),
    }
