from agents.stsa import stsa_agent
from schemas.stsa import STSAInput

def test_stsa_agent():
    stsa_input = STSAInput(
        updated_shariah_section={
            "clause_id": "FAS4.DM1",
            "text": "The diminishing Musharaka contract must clearly define the equity reduction schedule...",
            "reference": "AAOIFI SS 12"
        },
        updated_accounting_section={
            "clause_id": "FAS4.DM.ACC1",
            "text": "The accounting for diminishing Musharaka must recognize the gradual transfer of ownership as a series of separate transactions...",
            "reference": "FAS 4, FAS 32"
        },
        FAS="FAS 4",
        user_context="The user needs a fully updated FAS 4 for diminishing Musharaka in real estate funds.",
        knowledge_indexes=["fas_4"]
    )
    result = stsa_agent(stsa_input)
    print(result)
    assert isinstance(result.all_updated_sections, dict)
    assert isinstance(result.change_log, list)
    assert isinstance(result.references, list) 