from typing import TypedDict, List, Dict, Any, Optional

class FASAnalysisResult(TypedDict): # More specific type for the result
    analysis_summary: Dict[str, Any]
    identified_gaps: List[Dict[str, Any]]
    identified_similarities: List[Dict[str, Any]]

class State(TypedDict):
    """
    State class for managing the state of the agent.
    """
    messages: List[Dict[str, Any]] # For chat history if needed
    thoughts: List[str] # General thoughts or logs
    fas_analysis_result: Optional[FASAnalysisResult] # Store the structured output
    current_target_fas_id: Optional[str] # To keep track of what's being analyzed
    current_context: Optional[str] # To keep track of the context
    current_step: int
    max_steps: int
    completed: bool