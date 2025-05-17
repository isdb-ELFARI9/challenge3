import sqlite3
import json
from datetime import datetime

DATABASE_FILE = 'pipeline_runs.db'

def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_prompt TEXT NOT NULL,
                fas_number TEXT NOT NULL,
                run_data TEXT NOT NULL, -- Store JSON dump of relevant data
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def save_pipeline_run(user_prompt: str, fas_number: str, run_data: dict) -> int:
    """Saves the data from a pipeline run to the database."""
    init_db() # Ensure DB and table exist
    run_data_json = json.dumps(run_data)
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO pipeline_runs (user_prompt, fas_number, run_data) VALUES (?, ?, ?)',
            (user_prompt, fas_number, run_data_json)
        )
        conn.commit()
        return cursor.lastrowid # Return the ID of the new row

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
                "user_prompt": user_prompt,
                "fas_number": fas_number,
                **run_data # Merge the loaded JSON data
            }
        return None

# Initialize the database when the module is imported
init_db()