from pydantic import BaseModel, Field

class UIRIAInput(BaseModel):
    user_prompt: str = Field(..., description="Raw, unstructured user input describing the problem or context.")

class UIRIAOutput(BaseModel):
    context: str
    identified_FAS: str
    extracted_entities: list[str]
    user_intent: str