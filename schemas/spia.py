from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any

class SPIAInput(BaseModel):
    # gap_report: Union[str, List[Union[str, Dict[str, Any]]]] = Field(..., description="Gap report from FCIA.")
    gap_report: str = Field(..., description="Gap report from FCIA.")
    # FAS: Union[str, List[str]] = Field(..., description="Relevant FAS standard(s) to review.")
    FAS: str = Field(..., description="Relevant FAS standard(s) to review.")
    user_context: str = Field(..., description="User context for the case.")
    knowledge_indexes: List[str] = Field(..., description="Indexes to use from the knowledge base (should be only SS index).")

class SPIAOutput(BaseModel):
    shariah_solution: str
    updated_shariah_clauses: List[Dict[str, Any]]
    references: List[str] 