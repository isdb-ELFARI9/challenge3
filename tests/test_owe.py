from agents.owe import owe_agent

def test_owe_agent():
    user_prompt = (
        "We are seeing more use of diminishing Musharaka in real estate funds, "
        "but FAS 4 doesn't seem to cover shirkah al-ʿaqd structures. "
        "How should we treat partner contributions and exits?"
    )
    result = owe_agent(user_prompt)
    print(result)
    assert "document" in result
    assert "change_summary" in result
    assert "reasoning_trace" in result