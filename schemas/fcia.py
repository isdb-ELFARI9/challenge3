from pydantic import BaseModel, Field
from typing import List, Union

class FCIAInput(BaseModel):
    context: str = Field(..., description="Structured context from orchestrator.")
    FAS: Union[str, List[str]] = Field(..., description="Relevant FAS standard(s) to review.")
    knowledge_indexes: List[str] = Field(..., description="Indexes to use from the knowledge base.")

class FCIAOutput(BaseModel):
    identified_gaps: List[str]
    affected_clauses: List[str]
    user_context: str
    FAS_reference: Union[str, List[str]] 