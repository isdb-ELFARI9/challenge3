# AAOIFI Standards Enhancement API

This API provides an interface for AI-driven enhancement of AAOIFI Islamic finance standards.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

3. Run the API:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

## API Usage

| Method | Path                       | Purpose                                                             |
| ------ | -------------------------- | ------------------------------------------------------------------- |
| POST   | `/enhance-standard`        | Run the AAOIFI enhancement pipeline on a user prompt.               |
| GET    | `/pipeline-runs`           | List every stored pipeline run (most-recent first).                 |
| GET    | `/pipeline-run/{run_id}`   | Retrieve the full record—including FAS markdown—for one run.        |
| GET/HEAD | `/ping`                 | Lightweight health-check for uptime monitors.                       |
 example :

**Request Body:**
```json
{
    "prompt": "Your question or request about AAOIFI standards"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/enhance-standard" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "We are seeing more use of diminishing Musharaka in real estate funds, but FAS 4 doesn't seem to cover shirkah al-ʿaqd structures. How should we treat partner contributions and exits?"}'
```

**Response:**
The API returns a JSON object containing:
- `document`: The complete standards document
- `change_summary`: A user-friendly summary of changes
- `reasoning_trace`: The full reasoning process

## Tests
there is a large suit of pytest tests that test all agents you can run them with
```bash
pytest .|tests|name_of_test
```

Focused test: Refinement Agent

The Refinement Agent (FRA) reviews specific changes based on feedback and reasoning traces.
```bash
python tests/test_fra_agent.py
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 
