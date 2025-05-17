import asyncio
import pytest
from agents.owe import owe_agent
import json
from datetime import datetime

def test_owe_agent():
    # Test data
    user_prompt = (
        "We are seeing more use of diminishing Musharaka in real estate funds, "
        "but FAS 4 doesn't seem to cover shirkah al-ʿaqd structures. "
        "How should we treat partner contributions and exits?"
    )
    
    # Run the pipeline
    result = asyncio.run(owe_agent(user_prompt))
    
    # Basic assertions
    assert "diff" in result
    assert "change_summary" in result
    assert "reasoning_trace" in result
    assert "change_statistics" in result
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/owe_test_result_{timestamp}.json"
    
    # Ensure test_results directory exists
    import os
    os.makedirs("test_results", exist_ok=True)
    
    # Write detailed results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=lambda o: o.__dict__)


    
    print(f"\nTest results written to: {output_file}")
    print("\nKey sections to review:")
    print("1. document: The complete updated standard")
    print("2. change_summary: User-friendly summary of changes")
    print("3. reasoning_trace: Detailed analysis from each agent:")
    print("   - uiria: User input analysis")
    print("   - fcia: Financial context analysis")
    print("   - spia: Shariah principles analysis")
    print("   - arda: Accounting rules analysis")
    print("   - stsa: Standards text analysis")
    print("4. old_outputs: Technical details of changes:")
    print("   - updated_fas_document: Dictionary of all modified sections")
    print("   - change_log: List of specific changes made")
    print("   - references: List of sources used for changes")

if __name__ == "__main__":
    test_owe_agent()