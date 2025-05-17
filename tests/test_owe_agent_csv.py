import asyncio
import pytest
import json
import csv
import os
from datetime import datetime
from agents.owe import owe_agent

def test_owe_agent_from_csv():
    """
    Test the OWE agent using a prompt constructed from the first entry in a CSV file.
    The CSV should have columns for Title and Summary (and optionally Resources).
    """
    # Path to the CSV file
    csv_file = "..\markdown_extract.csv"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Create test CSV if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Summary", "Resources"])
            writer.writerow([
                "Budget reconciliation bill fails to move out of Budget Committee",
                "On May 16, 2025, a significant budget reconciliation bill in the United States failed to advance from the Budget Committee. The bill, which included tax provisions recently approved by the House Ways and Means Committee, was blocked by a coalition of Republican fiscal conservatives and Democrats. The primary concerns cited were the bill's overall cost and its projected impact on federal deficits. This development signals ongoing political gridlock in U.S. fiscal policy and raises uncertainty over the future of proposed tax changes and extensions, including those related to the Tax Cuts and Jobs Act (TCJA).",
                "https://www.journalofaccountancy.com/"
            ])
    
    # Read the first entry from the CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use the first row and break
            title = row.get("Title", "")
            summary = row.get("Summary", "")
            break
    
    # Construct the prompt from title and summary
    user_prompt = f"{title}: {summary}"
    
    print(f"Using constructed prompt: {user_prompt}")
    
    # Run the pipeline
    result = asyncio.run(owe_agent(user_prompt))
    
    # Basic assertions
    assert "diff" in result
    assert "change_summary" in result
    assert "reasoning_trace" in result
    assert "change_statistics" in result
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/owe_test_csv_result_{timestamp}.json"
    
    # Ensure test_results directory exists
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
    test_owe_agent_from_csv()