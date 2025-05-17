from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ChangeRecord(BaseModel):
    old_text: str = Field(..., description="Original text from the old FAS version")
    new_text: str = Field(..., description="Updated text in the new FAS version")
    justification: str = Field(..., description="Strong justification for the change")
    section_id: str = Field(..., description="Identifier of the section where change occurred")
    change_type: str = Field(..., description="Type of change: addition, deletion, modification")

    def to_dict(self) -> Dict[str, str]:
        """Convert the ChangeRecord to a dictionary."""
        return {
            "old_paragraph": self.old_text,
            "new_paragraph": self.new_text,
            "justification": self.justification,
            "section": self.section_id,
            "type": self.change_type
        }

class FASDiffInput(BaseModel):
    new_fas_markdown: str = Field(..., description="Content of the new FAS markdown file")
    fas_number: str = Field(..., description="FAS number being analyzed")
    context: str = Field(..., description="Context about why changes were made")
    multi_agent_reasoning: Dict[str, Any] = Field(default={}, description="Reasoning trace from all agents")

class FASDiffOutput(BaseModel):
    changes: List[ChangeRecord] = Field(..., description="List of all changes with justifications")
    key_changes_summary: str = Field(..., description="Summary of key changes for markdown")
    change_statistics: Dict[str, int] = Field(..., description="Statistics about types of changes")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the FASDiffOutput to a dictionary."""
        return {
            "changes": [change.to_dict() for change in self.changes],
            "key_changes_summary": self.key_changes_summary,
            "change_statistics": self.change_statistics
        } 