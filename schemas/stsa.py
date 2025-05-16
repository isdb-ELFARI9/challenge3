from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any

class STSAInput(BaseModel):
    updated_shariah_section: Union[str, Dict[str, Any]] = Field(..., description="Updated Shariah section for the FAS.")
    updated_accounting_section: Union[str, Dict[str, Any]] = Field(..., description="Updated accounting section for the FAS.")
    FAS: Union[str, List[str]] = Field(..., description="Relevant FAS standard(s) to review.")
    user_context: str = Field(..., description="User context for the case.")
    knowledge_indexes: List[str] = Field(..., description="Indexes to use from the knowledge base (should be only FAS index).")

class STSAOutput(BaseModel):
    all_updated_sections: Dict[str, Any]
    original_sections: Dict[str, Any] = Field(..., description="Original sections from the FAS before updates")
    change_log: List[str]
    references: List[str] 