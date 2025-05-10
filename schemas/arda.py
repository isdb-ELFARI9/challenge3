from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any

class ARDAInput(BaseModel):
    shariah_update: Union[str, Dict[str, Any]] = Field(..., description="Shariah-compliant process/solution from SPIA.")
    FAS: Union[str, List[str]] = Field(..., description="Relevant FAS standard(s) to review.")
    user_context: str = Field(..., description="User context for the case.")
    knowledge_indexes: List[str] = Field(..., description="Indexes to use from the knowledge base (should be only FAS index).")

class ARDAOutput(BaseModel):
    updated_accounting_clauses: List[Dict[str, Any]]
    rationale: str
    references: List[str] 