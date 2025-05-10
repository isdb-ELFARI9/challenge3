from agents.fcia import fcia_agent
from schemas.fcia import FCIAInput

def test_fcia_agent():
    fcia_input = FCIAInput(
        context="The user is facing ambiguity in applying FAS 4 to diminishing Musharaka structures, especially shirkah al-ʿaqd in real estate funds.",
        FAS="FAS 4",
        knowledge_indexes=["FAS_index", "SS_index"]
    )
    result = fcia_agent(fcia_input)
    print("result of the agent fcia :",result)
    assert isinstance(result.identified_gaps, list)
    for gap in result.identified_gaps:
        assert isinstance(gap, dict)
    assert isinstance(result.affected_clauses, list)
    assert isinstance(result.user_context, str)
    assert isinstance(result.FAS_reference, (str, list)) 