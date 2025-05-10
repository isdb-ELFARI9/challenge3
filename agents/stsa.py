from schemas.stsa import STSAInput, STSAOutput
from utils.llm import call_gemini_llm
import os
import json
from pinecone import Pinecone
import openai

# Pinecone configuration from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_FAS_NAME = os.getenv("PINECONE_INDEX_FAS")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for required environment variables
required_vars = [
    PINECONE_API_KEY, PINECONE_ENVIRONMENT,
    PINECONE_INDEX_FAS_NAME, OPENAI_API_KEY
]
if any(v is None for v in required_vars):
    raise ValueError("One or more required environment variables are not set. Please check your .env file.")

# Initialize Pinecone client (new SDK)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_fas = pc.Index(PINECONE_INDEX_FAS_NAME)

# Embedding function using OpenAI
openai.api_key = OPENAI_API_KEY
def get_embedding(text: str) -> list:
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def get_fas_namespace(fas: str) -> str:
    mapping = {
        '4': 'fas_4',
        '7': 'fas_7',
        '10': 'fas_10',
        '28': 'fas_28',
        '32': 'fas_32',
    }
    if isinstance(fas, str) and fas.lower().startswith('fas'):
        num = fas.split()[-1]
        return mapping.get(num, None)
    elif isinstance(fas, list):
        return [mapping.get(f.split()[-1], None) for f in fas if f.lower().startswith('fas')]
    return None

def retrieve_knowledge_from_pinecone_fas(query: str, fas_namespace: str) -> str:
    embedding = get_embedding(query)
    # Query Pinecone for the top 2 matches in the correct namespace
    result = index_fas.query(vector=embedding, top_k=2, namespace=fas_namespace, include_metadata=True)
    if result and result.matches:
        # Concatenate the text of the top matches
        return '\n'.join([m.metadata.get('text', '') for m in result.matches])
    return '[No relevant FAS document found]'

def build_stsa_prompt(updated_shariah_section, updated_accounting_section, FAS, user_context, knowledge: str) -> str:
    return (
        f"You are the Standard Text & Structure Agent (STSA) for an Islamic finance standards review system.\n\n"
        f"Role: You are an expert standards editor.\n\n"
        f"Your job is to:\n"
        f"- Integrate the updated Shariah and accounting sections into the FAS.\n"
        f"- Ensure clarity, consistency, and logical structure throughout the standard.\n"
        f"- Use the provided FAS knowledge base to ensure alignment with existing sections and terminology.\n"
        f"- Output your findings in a structured JSON format with the following fields:\n"
        f"  - all_updated_sections: a dictionary with all updated FAS sections (keys: 'shariah', 'accounting', and any others you update).\n"
        f"  - change_log: a list of concise descriptions of what was changed or added.\n"
        f"  - references: a list of FAS or other standards used.\n\n"
        f"Example output:\n"
        f"{{\n"
        f"  \"all_updated_sections\": {{\n"
        f"    \"shariah\": {{...}},\n"
        f"    \"accounting\": {{...}},\n"
        f"    \"disclosure\": {{...}}\n"
        f"  }},\n"
        f"  \"change_log\": [\n"
        f"    \"Added new Shariah section for diminishing Musharaka.\",\n"
        f"    \"Revised accounting section to clarify asset transfer.\"\n"
        f"  ],\n"
        f"  \"references\": [\"FAS 4\", \"FAS 32\"]\n"
        f"}}\n\n"
        f"Knowledge base context:\n{knowledge}\n\n"
        f"Updated Shariah section:\n{updated_shariah_section}\n\n"
        f"Updated accounting section:\n{updated_accounting_section}\n\n"
        f"User context:\n{user_context}\n\n"
        f"FAS to review:\n{FAS}\n\n"
        f"Respond ONLY with the JSON object."
    )

def stsa_agent(stsa_input: STSAInput) -> STSAOutput:
    fas_namespace = get_fas_namespace(stsa_input.FAS)
    # Use both updated sections as the query for best context
    query = f"{stsa_input.updated_shariah_section}\n{stsa_input.updated_accounting_section}"
    knowledge = retrieve_knowledge_from_pinecone_fas(query, fas_namespace)
    prompt = build_stsa_prompt(stsa_input.updated_shariah_section, stsa_input.updated_accounting_section, stsa_input.FAS, stsa_input.user_context, knowledge)
    response = call_gemini_llm(prompt)
    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    data = json.loads(response)
    return STSAOutput(**data) 