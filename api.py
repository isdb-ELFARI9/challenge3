import asyncio
import json
import os
import sqlite3
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.owe import owe_agent

# Database configuration
DATABASE_FILE = "pipeline_runs.db"

def init_db():
    """Initialize the database if it doesn't exist."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_prompt TEXT NOT NULL,
            fas_number TEXT NOT NULL,
            run_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()

def get_pipeline_run(run_id: int) -> dict | None:
    """Retrieves data for a specific pipeline run from the database."""
    init_db() # Ensure DB exists before trying to connect
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT user_prompt, fas_number, run_data FROM pipeline_runs WHERE run_id = ?', (run_id,))
        row = cursor.fetchone()
        if row:
            user_prompt, fas_number, run_data_json = row
            run_data = json.loads(run_data_json)
            # Reconstruct the data structure
            return {
                "run_id": run_id,
                "run_data": run_data,
                **run_data # Merge the loaded JSON data
            }
        return None

def get_all_pipeline_runs() -> list:
    """Retrieves all pipeline runs from the database."""
    init_db() # Ensure DB exists before trying to connect
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT run_id, user_prompt, fas_number, run_data, timestamp FROM pipeline_runs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        results = []
        for row in rows:
            run_id,user_prompt,fas_number, run_data_json , timestamp = row
            run_data = json.loads(run_data_json)
            # Only include key summary information to keep response size manageable
            result={
                "run_id": run_id,
                "run_data": run_data,
            }
            try:
                fas_old_file = run_data.get("reasoning_trace", {}).get("fas_gaps", {}).get("overall_verdict", {}).get("fas_to_update", [])[0]
                fas_old_file_path = f"fas_markdowns\\old\\{fas_old_file}.md"
                
                # Check if file exists before attempting to read
                if os.path.exists(fas_old_file_path):
                    with open(fas_old_file_path, "r", encoding="utf-8") as file:
                        fas_old_file_content = file.read()
                    result["fas_old_file"] = fas_old_file_content
                else:
                    result["fas_old_file"] = f"Could not find file: {fas_old_file}.md"
            except (KeyError, IndexError, TypeError) as e:
                # Handle the case where the required keys don't exist
                result["fas_old_file"] = f"Error retrieving FAS file: {str(e)}"
            
            results.append(result)
        return results


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


@app.get("/pipeline-runs")
async def get_all_runs():
    try:
        # Get all pipeline runs from the database
        results = get_all_pipeline_runs()
        
        if not results:
            return {"message": "No pipeline runs found", "runs": []}
        
        return {"runs": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving pipeline runs: {str(e)}")

@app.get("/pipeline-run/{run_id}")
async def get_run(run_id: int = Path(..., description="The ID of the pipeline run to retrieve")):
    try:
        # Get the run data from the database
        result = get_pipeline_run(run_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Pipeline run with ID {run_id} not found")
        
        # Extract FAS file information and add content
        try:
            fas_old_file = result["reasoning_trace"]["fas_gaps"]["overall_verdict"]["fas_to_update"][0]
            fas_old_file_path = f"fas_markdowns\\old\\{fas_old_file}.md"
            
            # Check if file exists before attempting to read
            if os.path.exists(fas_old_file_path):
                with open(fas_old_file_path, "r", encoding="utf-8") as file:
                    fas_old_file_content = file.read()
                result["fas_old_file"] = fas_old_file_content
            else:
                result["fas_old_file"] = f"Could not find file: {fas_old_file}.md"
        except (KeyError, IndexError) as e:
            # Handle the case where the required keys don't exist in the result
            result["fas_old_file"] = f"Error retrieving FAS file: {str(e)}"
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving pipeline run: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 