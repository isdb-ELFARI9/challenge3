from schemas.spia import SPIAInput, SPIAOutput
from utils.llm import call_gemini_llm
import os
import json
from pinecone import Pinecone
import openai

# Pinecone configuration from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_SS_NAME = os.getenv("PINECONE_INDEX_SS")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for required environment variables
required_vars = [
    PINECONE_API_KEY, PINECONE_ENVIRONMENT,
    PINECONE_INDEX_SS_NAME, OPENAI_API_KEY
]
if any(v is None for v in required_vars):
    raise ValueError("One or more required environment variables are not set. Please check your .env file.")

# Initialize Pinecone client (new SDK)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_ss = pc.Index(PINECONE_INDEX_SS_NAME)

# Embedding function using OpenAI
openai.api_key = OPENAI_API_KEY
def get_embedding(text: str) -> list:
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def retrieve_knowledge_from_pinecone_ss(query: str, ss_namespace: str) -> str:
    embedding = get_embedding(query)
    # Query Pinecone for the top 2 matches in the correct namespace
    result = index_ss.query(vector=embedding, top_k=2, namespace=ss_namespace, include_metadata=True)
    if result and result.matches:
        # Concatenate the text of the top matches
        return '\n'.join([m.metadata.get('text', '') for m in result.matches])
    return '[No relevant Shariah document found]'

def build_spia_prompt(gap_report, FAS, user_context, knowledge: str) -> str:
    return (
        f"You are the Shariah Principles Integration Agent (SPIA) for an Islamic finance standards review system.\n\n"
        f"Your job is to:\n"
        f"- Analyze the gap report and user context in light of the relevant FAS.\n"
        f"- Use the provided Shariah standards knowledge base to propose a Shariah-compliant solution to fill the identified gap.\n"
        f"- Suggest updated or new Shariah clauses for the FAS.\n"
        f"- Reference the specific Shariah standards or fatwas used.\n"
        f"- Output your findings in a structured JSON format with the following fields:\n"
        f"  - shariah_solution: a concise description of the Shariah-compliant process or solution.\n"
        f"  - updated_shariah_clauses: a list of new or revised Shariah clauses, each as a JSON object with at least the following fields:\n"
        f"      - clause_id: a unique identifier for the clause (e.g., 'FAS4.DM1')\n"
        f"      - text: the full text of the clause\n"
        f"      - reference: (optional) the Shariah standard or fatwa supporting this clause\n"
        f"  - references: a list of Shariah standards, fatwas, or sources used.\n\n"
        f"Example output:\n"
        f"{{\n"
        f"  \"shariah_solution\": \"A Shariah-compliant process for diminishing Musharaka in real estate funds...\",\n"
        f"  \"updated_shariah_clauses\": [\n"
        f"    {{\n"
        f"      \"clause_id\": \"FAS4.DM1\",\n"
        f"      \"text\": \"The diminishing Musharaka contract must clearly define the equity reduction schedule...\",\n"
        f"      \"reference\": \"AAOIFI SS 12\"\n"
        f"    }}\n"
        f"  ],\n"
        f"  \"references\": [\"AAOIFI SS 12\", \"Fatwa 123/2022\"]\n"
        f"}}\n\n"
        f"Knowledge base context:\n{knowledge}\n\n"
        f"Gap report:\n{gap_report}\n\n"
        f"User context:\n{user_context}\n\n"
        f"FAS to review:\n{FAS}\n\n"
        f"Respond ONLY with the JSON object."
    )

def spia_agent(spia_input: SPIAInput) -> SPIAOutput:
    # For this agent, always use the SS index and the namespace should be based on the most relevant SS (simulate with first in knowledge_indexes)
    ss_namespace = spia_input.knowledge_indexes[0] if spia_input.knowledge_indexes else None
    knowledge = retrieve_knowledge_from_pinecone_ss(str(spia_input.gap_report), ss_namespace)
    prompt = build_spia_prompt(spia_input.gap_report, spia_input.FAS, spia_input.user_context, knowledge)
    response = call_gemini_llm(prompt)
    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    data = json.loads(response)
    return SPIAOutput(**data) 