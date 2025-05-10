from schemas.arda import ARDAInput, ARDAOutput
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
    print("result of the query pinecone fas :",result)
    if result and result.matches:
        # Concatenate the text of the top matches
        return '\n'.join([m.metadata.get('text', '') for m in result.matches])
    return '[No relevant FAS document found]'

def build_arda_prompt(shariah_update, FAS, user_context, knowledge: str) -> str:
    return (
        f"You are the Accounting Rules Definition Agent (ARDA) for an Islamic finance standards review system.\n\n"
        f"Role: You are an expert Islamic finance accountant and standards developer.\n\n"
        f"Your job is to:\n"
        f"- Analyze the Shariah-compliant process/solution and user context in light of the relevant FAS.\n"
        f"- Use the provided FAS knowledge base to propose updated or new accounting rules/clauses for the FAS.\n"
        f"- For each clause, provide a clear rationale and reference to relevant standards.\n"
        f"- Output your findings in a structured JSON format with the following fields:\n"
        f"  - updated_accounting_clauses: a list of new or revised accounting clauses, each as a JSON object with at least the following fields:\n"
        f"      - clause_id: a unique identifier for the clause (e.g., 'FAS4.DM.ACC1')\n"
        f"      - text: the full text of the clause\n"
        f"      - reference: (optional) the FAS or other standard supporting this clause\n"
        f"  - rationale: a concise justification for the changes, including chain-of-thought reasoning.\n"
        f"  - references: a list of FAS or other standards used.\n\n"
        f"Example output:\n"
        f"{{\n"
        f"  \"updated_accounting_clauses\": [\n"
        f"    {{\n"
        f"      \"clause_id\": \"FAS4.DM.ACC1\",\n"
        f"      \"text\": \"The accounting for diminishing Musharaka must recognize the gradual transfer of ownership as a series of separate transactions...\",\n"
        f"      \"reference\": \"FAS 4, FAS 32\"\n"
        f"    }}\n"
        f"  ],\n"
        f"  \"rationale\": \"The new clause clarifies the accounting for diminishing Musharaka in line with the Shariah solution and ensures consistency with FAS 4 and FAS 32.\",\n"
        f"  \"references\": [\"FAS 4\", \"FAS 32\"]\n"
        f"}}\n\n"
        f"Knowledge base context:\n{knowledge}\n\n"
        f"Shariah update:\n{shariah_update}\n\n"
        f"User context:\n{user_context}\n\n"
        f"FAS to review:\n{FAS}\n\n"
        f"Respond ONLY with the JSON object."
    )

def arda_agent(arda_input: ARDAInput) -> ARDAOutput:
    fas_namespace = get_fas_namespace(arda_input.FAS)
    knowledge = retrieve_knowledge_from_pinecone_fas(str(arda_input.shariah_update), fas_namespace)
    prompt = build_arda_prompt(arda_input.shariah_update, arda_input.FAS, arda_input.user_context, knowledge)
    response = call_gemini_llm(prompt)
    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    data = json.loads(response)
    return ARDAOutput(**data) 