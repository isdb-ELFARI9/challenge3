import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from fas_gaps_similarities_identifier_agent.data_models import State, FASAnalysisResult
from fas_gaps_similarities_identifier_agent.llm import get_llm, LLMProvider
from fas_gaps_similarities_identifier_agent.fas_gaps_and_similarities_detector_agent import fas_gaps_and_similarities_detector_agent

async def run_parallel_fas_agents(
    context: str,
    target_fas_ids: List[str],
    llm_provider: LLMProvider = "openai",
    output_dir: str = "./outputs"
) -> Dict[str, State]:
    """
    Run multiple FAS agents in parallel for different target FAS IDs
    
    Args:
        context: The user input/context to analyze
        target_fas_ids: List of FAS IDs to analyze against
        llm_provider: The LLM provider to use
        output_dir: Directory to save the output files
        
    Returns:
        Dictionary mapping FAS IDs to their corresponding final states
    """
    print(f"Running parallel analysis for {len(target_fas_ids)} FAS standards: {', '.join(target_fas_ids)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tasks for each FAS agent
    tasks = []
    for fas_id in target_fas_ids:
        # Initialize the base state with user input
        initial_state = State(
            messages=[{"role": "human", "content": context}],
            thoughts=[],
            fas_analysis_result=None,
            current_target_fas_id=fas_id,
            current_context=context,
            current_step=0,
            max_steps=1,
            completed=False
        )
        
        # Create output file path
        output_file = os.path.join(output_dir, f"{fas_id}_analysis.json")
        
        # Create coroutine for this agent
        task = asyncio.create_task(
            run_fas_agent(initial_state, fas_id, llm_provider, output_file)
        )
        tasks.append((fas_id, task))
    
    # Wait for all tasks to complete concurrently
    fas_ids = [fas_id for fas_id, _ in tasks]
    task_objects = [task for _, task in tasks]
    
    # Using gather to wait for all tasks to complete simultaneously
    task_results = await asyncio.gather(*task_objects, return_exceptions=True)
    
    # Process results
    results = {}
    for fas_id, result in zip(fas_ids, task_results):
        if isinstance(result, Exception):
            print(f"✗ Analysis for {fas_id} generated an exception: {result}")
            # Create an error state
            error_state = State(
                messages=[{"role": "human", "content": context}],
                thoughts=[f"Error occurred during analysis: {str(result)}"],
                fas_analysis_result=None,
                current_target_fas_id=fas_id,
                current_context=context,
                current_step=0,
                max_steps=1,
                completed=False
            )
            results[fas_id] = error_state
        else:
            results[fas_id] = result
            print(f"✓ Completed analysis for {fas_id}")
    
    return results

async def run_fas_agent(
    initial_state: State, 
    target_fas_id: str, 
    llm_provider: LLMProvider, 
    output_file: str
) -> State:
    """Run a single FAS agent as an async operation"""
    # Make a deep copy to avoid shared state issues
    agent_state = State(
        messages=initial_state["messages"].copy(),
        thoughts=initial_state["thoughts"].copy(),
        fas_analysis_result=initial_state.get("fas_analysis_result"),
        current_target_fas_id=target_fas_id,
        current_context=initial_state.get("current_context"),
        current_step=initial_state.get("current_step", 0),
        max_steps=initial_state.get("max_steps", 1),
        completed=initial_state.get("completed", False)
    )
    
    # Run the agent (wrapping the sync function in an async context)
    return fas_gaps_and_similarities_detector_agent(
        agent_state, 
        target_fas_id=target_fas_id, 
        llm_provider=llm_provider,
        output_file=output_file
    )
