import asyncio
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
        result = await owe_agent(user_prompt.prompt)
        #include in the result the content of markdown fas file that are in fas_markdowns\old and the naming of files is for example fas_i.md and include in in a field of result called fas_old_file , to get fas_i you should see result.reasoning_trace.fas_gaps.overall_verdict.fas_to_update[0]
        print("from the api",result)
        fas_old_file = result["reasoning_trace"]["fas_gaps"]["overall_verdict"]["fas_to_update"][0]
        print(fas_old_file)
        fas_old_file_path = f"fas_markdowns\old\{fas_old_file}.md"
        print(fas_old_file_path)
        with open(fas_old_file_path, "r") as file:
            fas_old_file_content = file.read()
            print(fas_old_file_content)
        result["fas_old_file"] = fas_old_file_content
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 