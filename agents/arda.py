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
    if result and result.matches:
        # Concatenate the text of the top matches
        return '\n'.join([m.metadata.get('text', '') for m in result.matches])
    return '[No relevant FAS document found]'

def build_arda_prompt(shariah_update, FAS, user_context, knowledge: str) -> str:
    return (
        "You are the Accounting Rules Definition Agent (ARDA) for an Islamic finance standards review system.\n\n"
        "Role: You are a senior Islamic finance accountant and FAS standards developer.\n\n"
        "Your task is to:\n"
        "1. Analyze the Shariah-compliant process/solution and user context in light of the relevant FAS.\n"
        "2. Consult the provided FAS knowledge base for relevant accounting principles, rules, and precedents.\n"
        "3. Propose updated or new accounting rules/clauses for the FAS, ensuring:\n"
        "   - Alignment with the Shariah update\n"
        "   - Consistency with FAS and, where relevant, IFRS or AAOIFI standards\n"
        "   - Practical applicability for IFIs\n"
        "4. For each clause, provide:\n"
        "   - clause_id: a unique identifier (e.g., 'FAS4.DM.ACC1')\n"
        "   - text: the full text of the clause\n"
        "   - reference: (optional) the FAS or other standard supporting this clause\n"
        "5. For the rationale, explain:\n"
        "   - Why each change is needed\n"
        "   - How it addresses the gap and aligns with Shariah and accounting best practices\n"
        "6. List all references used.\n\n"
        "Output your findings in the following JSON format:\n"
        "{\n"
        "  \"updated_accounting_clauses\": [\n"
        "    {\n"
        "      \"clause_id\": \"FAS4.DM.ACC1\",\n"
        "      \"text\": \"The accounting for diminishing Musharaka must recognize the gradual transfer of ownership as a series of separate transactions...\",\n"
        "      \"reference\": \"FAS 4, FAS 32\"\n"
        "    }\n"
        "  ],\n"
        "  \"rationale\": \"The new clause clarifies the accounting for diminishing Musharaka in line with the Shariah solution and ensures consistency with FAS 4 and FAS 32. This addresses the gap in equity transfer and profit allocation.\",\n"
        "  \"references\": [\"FAS 4\", \"FAS 32\", \"IFRS 15\"]\n"
        "}\n\n"
        f"Knowledge base context:\n{knowledge}\n\n"
        f"Shariah update:\n{shariah_update}\n\n"
        f"User context:\n{user_context}\n\n"
        f"FAS to review:\n{FAS}\n\n"
        "Respond ONLY with the JSON object."
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