from agents.spia import spia_agent
from schemas.spia import SPIAInput

def test_spia_agent():
    spia_input = SPIAInput(
        gap_report=[
            "FAS 4 lacks guidance on diminishing Musharaka (shirkah al-ʿaqd) in real estate funds.",
            "Ambiguity in profit allocation and asset transfer for these structures."
        ],
        FAS="FAS 4",
        user_context="The user needs a Shariah-compliant process for diminishing Musharaka in real estate funds.",
        knowledge_indexes=["SS12"]
    )
    result = spia_agent(spia_input)
    print(result)
    assert isinstance(result.shariah_solution, str)
    assert isinstance(result.updated_shariah_clauses, list)
    for clause in result.updated_shariah_clauses:
        assert isinstance(clause, dict)
    assert isinstance(result.references, list) 