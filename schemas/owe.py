from pydantic import BaseModel
from typing import Dict, Any, List

class OWEInput(BaseModel):
    """Input model for the OWE agent."""
    user_prompt: str

class OWEOutput(BaseModel):
    """Output model for the OWE agent."""
    document: str
    change_summary: str
    reasoning_trace: Dict[str, Any]
    old_outputs: Dict[str, Any] 