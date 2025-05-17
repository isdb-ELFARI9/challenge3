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
        f"""You are the Accounting Rules Definition Agent (ARDA) for an Islamic finance standards review system.

        Role: You are a senior Islamic finance accountant and FAS standards developer with extensive experience in translating Shariah principles and modern financial practices into clear, practical accounting standards.

        Your task is to:
        1. Analyze the proposed Shariah-compliant solution/update and the user context (including the potential implications for Islamic finance activities) in light of the relevant FAS.
        2. Consult the provided FAS knowledge base, other relevant AAOIFI Accounting Standards, and potentially IFRS where applicable to identify existing principles, rules, or precedents relevant to the accounting treatment of the activities implied by the Shariah solution and user context (e.g., accounting for digital assets, specific types of trading or investment).
        3. Propose clear, detailed, and practical updated or new accounting rules/clauses for the FAS. Ensure these rules are:
            - In direct alignment with the Shariah update provided by SPIA.
            - Consistent with existing FAS and other relevant AAOIFI Accounting Standards.
            - Consider relevant aspects from IFRS where they don't conflict with Shariah or AAOIFI principles.
            - Practically applicable for Islamic Financial Institutions.
        4. For each proposed accounting clause, provide:
            - "clause_id": a unique identifier (e.g., 'FAS[X].[GapID].ACC[Y]')
            - "text": the full, precisely worded clause suitable for inclusion in a standard, including recognition, measurement, presentation, and disclosure requirements as appropriate.
            - "reference": (Optional, but highly recommended) cite the specific FAS, AAOIFI AS, or IFRS section/clause that provides a basis or analogy for this rule.
        5. Provide a detailed rationale for *each* proposed clause and for the overall accounting approach. Explain the accounting logic, explicitly linking:
            - How the accounting rule implements the Shariah principle/solution.
            - How it addresses the specific accounting gap identified by FCIA related to the potential activities.
            - Its consistency with existing accounting frameworks (FAS, AAOIFI AS, relevant IFRS).
            - Any specific practical considerations for IFIs.
            - Cite relevant sources (FAS, AAOIFI AS, IFRS) within the rationale to support the accounting treatment.
        6. List all specific standards or documents referenced in the "references" array.

        Output your findings in the following JSON format:
        {{
        "updated_accounting_clauses": [
            {{
            "clause_id": "FAS4.Crypto.ACC1",
            "text": "...",
            "reference": "AAOIFI AS [Specific Relevant Standard]"
            }}
        ],
        "rationale": "...",
        "references": ["AAOIFI AS [Specific Investment Standard]", "FAS 4", "IFRS 9 (Conceptual Alignment)"]
        }}

        Knowledge base context:
        {knowledge}

        Shariah update:
        {shariah_update}

        User context:
        {user_context}

        FAS to review:
        {FAS}

        Respond ONLY with the JSON object.
        """
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