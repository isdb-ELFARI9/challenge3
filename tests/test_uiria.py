from agents.uiria import uiria_agent

def test_uiria_agent():
    user_prompt = (
        "We are seeing more use of diminishing Musharaka in real estate funds, "
        "but FAS 4 doesn't seem to cover shirkah al-ʿaqd structures. "
        "How should we treat partner contributions and exits?"
    )
    result = uiria_agent(user_prompt)
    print(result)
    assert isinstance(result.context, str)
    assert isinstance(result.identified_FAS, list)
    assert isinstance(result.extracted_entities, list)
    assert isinstance(result.user_intent, str)