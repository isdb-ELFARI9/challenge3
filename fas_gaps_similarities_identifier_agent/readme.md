# AAOIFI FAS Analysis and Synthesis Engine

## Overview

This project implements an advanced system for analyzing financial contexts, specifically those related to Islamic finance innovations like Digital Sukuk Al-Ijarah REITs (DSIRs), against existing AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions) Financial Accounting Standards (FAS).

The system leverages Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) to:
1.  Identify **gaps** where the provided context introduces accounting challenges not covered by specific FAS.
2.  Identify **similarities** where the context aligns with principles in specific FAS.
3.  Run these analyses in **parallel** for multiple FAS standards.
4.  **Synthesize** the findings from individual analyses into a consolidated verdict.
5.  Recommend whether existing FAS standards require **updates** or if a **new FAS standard** might be necessary.

The output includes detailed JSON reports for each analysis step and a final human-readable Markdown summary.

## Features

*   **Automated FAS Analysis:** Compares user-provided financial contexts against specified AAOIFI FAS standards.
*   **Gap & Similarity Detection:** Pinpoints discrepancies and alignments between the context and FAS.
*   **Parallel Processing:** Analyzes against multiple FAS standards concurrently for efficiency.
*   **Retrieval Augmented Generation (RAG):** Uses a Pinecone vector database to fetch relevant FAS knowledge dynamically, providing context to the LLM.
*   **LLM-Powered Synthesis:** Consolidates individual FAS analyses into a comprehensive final verdict and recommendations.
*   **Multi-LLM Support:** Configurable to use different LLM providers (e.g., OpenAI, Google Gemini).
*   **Structured Output:** Generates detailed JSON outputs for each analysis and the final synthesis.
*   **Human-Readable Reports:** Produces a final summary report in Markdown format.

## Architecture / Workflow

The system follows this general workflow:

1.  **User Input:** A financial context (e.g., description of a new financial instrument and its accounting challenges) is provided.
2.  **Parallel FAS Analysis:**
    *   For each target FAS standard (e.g., FAS 4, FAS 8):
        *   The `fas_gaps_and_similarities_detector_agent` is invoked.
        *   The `fas_retriever_agent` fetches relevant FAS document excerpts from a Pinecone vector database (RAG).
        *   The LLM analyzes the user input against the retrieved FAS knowledge and the target FAS ID.
        *   A structured JSON output (`fas_X_analysis.json`) is generated, detailing identified gaps, similarities, justifications, and scores.
3.  **Synthesis:**
    *   The `synthesizer_agent` receives the original user input and all individual `fas_X_analysis.json` files.
    *   The LLM processes this consolidated information.
    *   A final structured JSON output (`synthesis_result.json`) is generated, containing:
        *   An overall verdict.
        *   Recommendations for which FAS standards to update (if any).
        *   A decision on whether a new FAS standard is needed.
        *   Detailed justifications and chains of thought for these recommendations.
4.  **Formatted Output:**
    *   The `synthesis_result.json` is used to generate a human-readable Markdown report (`formatted_results.md`).

## Prerequisites

*   Python 3.8+
*   Access to a Pinecone account and an existing vector index populated with AAOIFI FAS documents.
*   API Keys for:
    *   OpenAI
    *   Google Generative AI (for Gemini models)
    *   Pinecone

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    langchain
    langchain-openai
    langchain-google-genai
    openai
    pinecone-client
    python-dotenv
    asyncio
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project and add your API keys and Pinecone details:
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIRONMENT="your_pinecone_environment"
    PINECONE_INDEX_FAS="your_pinecone_fas_index_name"
    ```

## How to Run

The main entry point for the workflow is `run_workflow.py`.

1.  **Configure the Workflow:**
    Open `run_workflow.py` and modify the following as needed:
    *   `user_input`: Provide the financial context you want to analyze. An example is already provided in the script.
    *   `fas_standards`: A list of FAS IDs to analyze against (e.g., `["fas_4", "fas_8", "fas_16"]`).
    *   `selected_provider`: Set to `"openai"` or `"gemini"`.
    *   `output_dir`: The directory where results will be saved (defaults to `./outputs`).

2.  **Execute the Workflow:**
    ```bash
    python run_workflow.py
    ```

3.  **View Results:**
    *   Individual FAS analysis JSON files (e.g., `fas_4_analysis.json`) will be saved in the `output_dir`.
    *   The consolidated synthesis JSON (`synthesis_result.json`) will be saved in `output_dir`.
    *   A human-readable Markdown summary (`formatted_results.md`) will be saved in `output_dir` and also printed to the console.

## Project Structure
├── config.py # Configuration like supported FAS lists
├── data_models.py # TypedDicts for data structures
├── fas_gaps_and_similarities_detector_agent.py # Agent for individual FAS analysis
├── fas_retriever_agent.py # Agent for RAG from Pinecone
├── llm.py # LLM provider interface
├── parallel_agent_runner.py # Manages parallel execution of FAS agents
├── run_workflow.py # Main script to execute the entire workflow
├── synthesizer.py # Agent for synthesizing results
├── outputs/ # Directory for generated output files
│ ├── fas_4_analysis.json # Example individual analysis output
│ ├── fas_8_analysis.json # Example individual analysis output
│ ├── fas_16_analysis.json # Example individual analysis output
│ ├── synthesis_result.json # Example synthesis JSON output
│ └── formatted_results.md # Example formatted Markdown report
├── .env # (User-created) For API keys and environment variables
└── requirements.txt # (User-created) Python dependencies



*Example input files (which are actually outputs of the system run) like `fas_4_analysis.json`, `formatted_results.md`, etc., are provided to illustrate the expected output structure.*

## Core Agents and Their Roles

*   **`fas_gaps_and_similarities_detector_agent.py`:**
    *   Takes a financial context and a specific `target_fas_id`.
    *   Utilizes `fas_retriever_agent` to fetch relevant FAS knowledge (RAG).
    *   Prompts an LLM to compare the context against the FAS knowledge.
    *   Outputs a structured JSON detailing identified gaps, similarities, justifications, references, and confidence scores.

*   **`fas_retriever_agent.py`:**
    *   Interfaces with a Pinecone vector database.
    *   Embeds search queries (typically FAS IDs or related concepts).
    *   Retrieves relevant document chunks from the FAS knowledge base stored in Pinecone.

*   **`parallel_agent_runner.py`:**
    *   Takes a list of `target_fas_ids`.
    *   Asynchronously runs multiple instances of the `fas_gaps_and_similarities_detector_agent`, one for each FAS ID.
    *   Collects and saves the individual JSON analysis results.

*   **`synthesizer.py`:**
    *   Takes the original user context and the collection of individual FAS analysis JSONs.
    *   Prompts an LLM to perform a meta-analysis of these results.
    *   Generates a comprehensive verdict on which FAS standards (if any) need updates, whether a new FAS is required, and provides detailed reasoning.
    *   Also includes a function (`format_synthesis_results`) to convert the synthesis JSON into a human-readable Markdown report.

## Input and Output

### Input

*   **User Context:** A string provided in `run_workflow.py` (variable `user_input`). This string describes the financial instrument, scenario, or accounting challenges to be analyzed.
*   **Target FAS IDs:** A list of strings in `run_workflow.py` (variable `fas_standards`) specifying which AAOIFI FAS standards to analyze against.

### Output

All output files are saved in the directory specified by `output_dir` (default: `./outputs/`).

*   **`fas_X_analysis.json`:** (e.g., `fas_4_analysis.json`)
    *   A JSON file generated for each target FAS standard.
    *   Contains:
        *   `analysis_summary`: Overall assessment and key metrics for that specific FAS.
        *   `identified_gaps`: A list of gaps, each with description, justification, references, chain of thought, and score.
        *   `identified_similarities`: A list of similarities, with similar details.
        *   `target_fas_id`: The FAS ID this analysis pertains to.

*   **`synthesis_result.json`:**
    *   A single JSON file consolidating findings from all individual FAS analyses.
    *   Contains:
        *   `overall_verdict`: Includes `fas_to_update` (list), `need_new_fas` (boolean), `overall_justification`, `overall_chain_of_thought`, and referenced gaps/similarities.
        *   `updated_fas_details`: A list of objects, each detailing why a specific FAS needs an update, with justification, chain of thought, and referenced gaps/similarities.
        *   `new_fas_details`: (Optional) Details if a new FAS is recommended, including justification, proposed scope, and influencing gaps.

*   **`formatted_results.md`:**
    *   A human-readable Markdown report summarizing the `synthesis_result.json`.
    *   Includes the overall verdict, justifications for updates, and details if a new FAS is proposed.

## Configuration

*   **`run_workflow.py`:**
    *   `user_input`: The main text to be analyzed.
    *   `fas_standards`: List of FAS IDs (e.g., "fas_4", "fas_8").
    *   `selected_provider`: LLM provider ("openai" or "gemini").
    *   `output_dir`: Directory for output files.
*   **`config.py`:**
    *   `supported_fas_list`: A `Literal` type defining FAS IDs, primarily for type hinting or potential validation. The actual list processed is determined by `fas_standards` in `run_workflow.py`.
*   **`.env` file:** (User-created)
    *   API keys and Pinecone connection details.

## Key Dependencies

*   **Langchain:** Framework for developing applications powered by language models.
*   **OpenAI Python Client:** For interacting with OpenAI APIs (GPT models, embeddings).
*   **Google Generative AI SDK:** For interacting with Gemini models.
*   **Pinecone Client:** For interacting with the Pinecone vector database.
*   **python-dotenv:** For managing environment variables from a `.env` file.
*   **asyncio:** For concurrent execution of FAS agents.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is unlicensed (or specify your preferred license, e.g., MIT License).