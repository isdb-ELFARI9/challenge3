import os
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from fas_gaps_similarities_identifier_agent.data_models import State
from fas_gaps_similarities_identifier_agent.parallel_agent_runner import run_parallel_fas_agents
from fas_gaps_similarities_identifier_agent.synthesizer import synthesize_results, format_synthesis_results
from fas_gaps_similarities_identifier_agent.llm import LLMProvider
from fas_gaps_similarities_identifier_agent.config import supported_fas_list

# Load environment variables
load_dotenv()

async def run_complete_workflow(
    user_input: str,
    target_fas_ids: List[str] = ["fas_4", "fas_8", "fas_16"],
    llm_provider: LLMProvider = "openai",
    output_dir: str = "./outputs"
) -> Dict[str, Any]:
    """
    Run the complete workflow with parallel FAS agents and synthesis
    
    Args:
        user_input: The user input/context to analyze
        target_fas_ids: List of FAS IDs to analyze against
        llm_provider: The LLM provider to use
        output_dir: Directory to save the output files
        
    Returns:
        Dictionary with results and final formatted output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Run parallel FAS agents
    fas_results = await run_parallel_fas_agents(
        user_input,
        target_fas_ids,
        llm_provider,
        output_dir
    )
    
    print(f"\nAll FAS analyses completed. Running synthesis...")
    
    # Step 2: Run the synthesizer
    try:
        synthesis_output_file = os.path.join(output_dir, "synthesis_result.json")
        synthesis_result = synthesize_results(
            user_input,
            fas_results,
            llm_provider,
            synthesis_output_file
        )
        
        # Step 3: Format the results for human-readable output
        # formatted_output = format_synthesis_results(synthesis_result, fas_results)
        formatted_output = format_synthesis_results(synthesis_result)
        
        # Save the formatted output to a file
        formatted_output_file = os.path.join(output_dir, "formatted_results.md")
        with open(formatted_output_file, "w", encoding="utf-8") as f:
            f.write(formatted_output)
        
        return {
            "fas_results": fas_results,
            "synthesis_result": synthesis_result,
            "formatted_output": formatted_output
        }
    except Exception as e:
        print(f"Error in synthesis process: {e}")
        return {
            "fas_results": fas_results,
            "error": str(e)
        }

async def main():
    """Main function to run the multi-FAS agent workflow."""
    print("Starting Multi-FAS Agent Workflow")
    
    # Configure the FAS standards to analyze
    fas_standards = ["fas_4"]
    
    # Set the LLM provider
    selected_provider = "openai"  # Change to "gemini" to use Google's model
    
    # Get user input - replace this with actual user input
    user_input = """
    Digital Sukuk Al-Ijarah REITs (DSIRs) represent a novel financial innovation in Islamic finance that 
    combines traditional real estate investment trusts (REITs) with blockchain technology. DSIRs are 
    digital tokens that represent fractional ownership in Shariah-compliant real estate assets, with 
    periodic distributions derived from rental income. The underlying assets are tangible and permissible 
    (halal), while the profit distribution mimics Mudarabah or Musharakah principles. 
    
    However, this innovation introduces several accounting challenges that require consideration:
    
    1. Valuation guidance is needed for these digital tokens, which may experience volatility in the secondary market.
    2. Income recognition timing and nature need clarity when involving digital token transactions.
    3. Impairment testing requirements should address both the property value and token market value.
    4. Custody concerns arise related to private keys and potential losses from cyber-attacks.
    5. ESG compliance reporting may become mandatory for these instruments.
    
    These issues require clear accounting standards to ensure consistent financial reporting across institutions 
    adopting this innovative instrument.
    """
    
    # Create output directory
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the workflow
    result = await run_complete_workflow(user_input, fas_standards, selected_provider, output_dir)
    
    # Print the final formatted output
    if "formatted_output" in result:
        print(f"\n=== FINAL RESPONSE ===\n")
        print(result["formatted_output"])
        print("\n=====================")
    else:
        print(f"An error occurred: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
