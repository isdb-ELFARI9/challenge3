from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.owe import owe_agent

app = FastAPI(
    title="AAOIFI Standards Enhancement API",
    description="API for AI-driven enhancement of AAOIFI Islamic finance standards",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class UserPrompt(BaseModel):
    prompt: str

@app.post("/enhance-standard")
async def enhance_standard(user_prompt: UserPrompt):
    try:
        result = owe_agent(user_prompt.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 