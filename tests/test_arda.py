from agents.arda import arda_agent
from schemas.arda import ARDAInput

def test_arda_agent():
    arda_input = ARDAInput(
        shariah_update={
            "shariah_solution": "A Shariah-compliant process for diminishing Musharaka in real estate funds...",
            "updated_shariah_clauses": [
                {
                    "clause_id": "FAS4.DM1",
                    "text": "The diminishing Musharaka contract must clearly define the equity reduction schedule...",
                    "reference": "AAOIFI SS 12"
                }
            ],
            "references": ["AAOIFI SS 12"]
        },
        FAS="FAS 4",
        user_context="The user needs accounting rules for diminishing Musharaka in real estate funds.",
        knowledge_indexes=["fas_4"]
    )
    result = arda_agent(arda_input)
    print(result)
    assert isinstance(result.updated_accounting_clauses, list)
    for clause in result.updated_accounting_clauses:
        assert isinstance(clause, dict)
    assert isinstance(result.rationale, str)
    assert isinstance(result.references, list) 