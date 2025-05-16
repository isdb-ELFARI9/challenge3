import pytest
from agents.fas_diff import fas_diff_agent, update_markdown_with_changes
from schemas.fas_diff import FASDiffInput

def test_fas_diff_agent():
    # Test data
    old_markdown = """
    # FAS 4: Musharaka
    
    ## Section 1: Introduction
    Musharaka is a partnership where profits are shared according to agreed ratios.
    
    ## Section 2: Types
    There are two types of Musharaka: permanent and diminishing.
    """
    
    new_markdown = """
    # FAS 4: Musharaka
    
    ## Section 1: Introduction
    Musharaka is a partnership where profits are shared according to agreed ratios and losses are shared according to capital contribution ratios.
    
    ## Section 2: Types
    There are three types of Musharaka: permanent, diminishing, and shirkah al-ʿaqd.
    """
    
    # Create input
    input_data = FASDiffInput(
        old_fas_markdown=old_markdown,
        new_fas_markdown=new_markdown,
        fas_number="4",
        context="Adding coverage for shirkah al-ʿaqd structures and clarifying loss sharing"
    )
    
    # Run the agent
    result = fas_diff_agent(input_data)
    
    # Basic assertions
    assert len(result.changes) > 0
    assert result.key_changes_summary
    assert result.change_statistics
    
    # Verify change records
    for change in result.changes:
        assert change.old_text
        assert change.new_text
        assert change.justification
        assert change.section_id
        assert change.change_type in ["addition", "deletion", "modification"]
    
    # Test markdown update
    updated_markdown = update_markdown_with_changes(new_markdown, result.key_changes_summary)
    assert "## Key Changes from Previous FAS Version" in updated_markdown
    assert result.key_changes_summary in updated_markdown

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 