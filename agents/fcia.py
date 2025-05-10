from schemas.fcia import FCIAInput, FCIAOutput
from utils.llm import call_gemini_llm
import os
import json
from pinecone import Pinecone
import openai

# Pinecone configuration from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_FAS_NAME = os.getenv("PINECONE_INDEX_FAS")
PINECONE_INDEX_SS_NAME = os.getenv("PINECONE_INDEX_SS")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for required environment variables
required_vars = [
    PINECONE_API_KEY, PINECONE_ENVIRONMENT,
    PINECONE_INDEX_FAS_NAME, PINECONE_INDEX_SS_NAME,
    OPENAI_API_KEY
]
if any(v is None for v in required_vars):
    raise ValueError("One or more required environment variables are not set. Please check your .env file.")

# Initialize Pinecone client (new SDK)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_fas = pc.Index(PINECONE_INDEX_FAS_NAME)
index_ss = pc.Index(PINECONE_INDEX_SS_NAME)

# Embedding function using OpenAI
openai.api_key = OPENAI_API_KEY
def get_embedding(text: str) -> list:
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def get_fas_namespace(fas: str) -> str:
    # Map FAS number to namespace
    mapping = {
        '4': 'fas_4',
        '7': 'fas_7',
        '10': 'fas_10',
        '28': 'fas_28',
        '32': 'fas_32',
    }
    # Extract number from FAS string (e.g., 'FAS 4' -> '4')
    if isinstance(fas, str) and fas.lower().startswith('fas'):
        num = fas.split()[-1]
        return mapping.get(num, None)
    elif isinstance(fas, list):
        # If multiple FAS, return all relevant namespaces
        return [mapping.get(f.split()[-1], None) for f in fas if f.lower().startswith('fas')]
    return None

# Placeholder for Pinecone vector DB retrieval
# In real use, replace this with actual Pinecone query logic
# For now, simulate retrieval from the correct namespace

def retrieve_knowledge_from_pinecone(query: str, fas_namespace: str) -> str:
    embedding = get_embedding(query)
    # Query Pinecone for the top 1 match in the correct namespace
    result = index_fas.query(vector=embedding, top_k=1, namespace=fas_namespace, include_metadata=True)
    # print("vector query result :",result)
    if result and result.matches:
        return result.matches[0].metadata.get('text', '[No relevant document found]')
    return '[No relevant document found]'

def build_fcia_prompt(context: str, FAS, knowledge: str) -> str:
    return (
        f"You are the FAS Contextualizer & Impact Assessor Agent (FCIA) for an Islamic finance standards review system.\n\n"
        f"Your job is to:\n"
        f"- Analyze the provided user context and the specified FAS standard(s).\n"
        f"- Identify any gaps, ambiguities, or missing coverage in the FAS with respect to the user context.\n"
        f"- Use the provided knowledge base information to support your analysis.\n"
        f"- Output your findings in a structured JSON format with the following fields:\n"
        f"  - identified_gaps: a list of gaps or issues found.\n"
        f"  - affected_clauses: a list of FAS clauses that are relevant or insufficient.\n"
        f"  - user_context: a summary of the user context.\n"
        f"  - FAS_reference: the FAS standard(s) reviewed.\n\n"
        f"Knowledge base context:\n{knowledge}\n\n"
        f"User context:\n{context}\n\n"
        f"FAS to review:\n{FAS}\n\n"
        f"Respond ONLY with the JSON object."
    )

def fcia_agent(fcia_input: FCIAInput) -> FCIAOutput:
    fas_namespace = get_fas_namespace(fcia_input.FAS)
    knowledge = retrieve_knowledge_from_pinecone(fcia_input.context, fas_namespace)
    prompt = build_fcia_prompt(fcia_input.context, fcia_input.FAS, knowledge)
    response = call_gemini_llm(prompt)
    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    data = json.loads(response)
    return FCIAOutput(**data) 