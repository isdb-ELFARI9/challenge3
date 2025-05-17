from schemas.fcia import FCIAInput, FCIAOutput
from utils.llm import call_gemini_llm
import os
import json
from pinecone import Pinecone
import openai

from dotenv import load_dotenv
load_dotenv()

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
        f"""You are the FAS Contextualizer & Impact Assessor Agent (FCIA) for an Islamic finance standards review system.

        Role: You are an expert in Islamic finance standards, specializing in interpreting real-world events/contexts and performing gap analysis against existing standards.

        Your task is to:
        1. Carefully read the user context, which may be a general event, news item, or specific scenario, and the provided FAS standard.
        2. Interpret the implications of the user context for potential Islamic finance activities, transactions, or instruments. For example, if the news is about crypto legalization, think about how Islamic Financial Institutions (IFIs) might potentially engage with crypto (e.g., trading, investment, financing).
        3. Summarize the user's context and its key potential implications for Islamic finance in 1-3 sentences.
        4. Review the FAS clauses (using the knowledge base and your expertise) and identify any areas that are ambiguous, missing, or outdated with respect to the potential Islamic finance activities arising from the user's context.
        5. For each identified potential gap:
            - Specify the affected FAS clause (by number or description, if possible).
            - Explain precisely what is missing or unclear, linking it directly to the potential Islamic finance activity identified in step 2.
            - Provide a detailed justification for *why* this is a significant gap in light of the user's context's implications and potential future activities for IFIs.
        6. Justify your findings with specific references to the FAS text, relevant general Islamic finance concepts related to the context, or information from the knowledge base.

        Output your findings in the following JSON format:
        {{
        "identified_gaps": [
            {{
            "clause": "FAS 4 - Clause 7",
            "issue": "No guidance on the accounting treatment for crypto assets acquired for trading purposes.",
            "justification": "Saudi Arabia legalizing crypto implies IFIs may engage in crypto trading. FAS 4, which covers investment accounting, lacks specific rules for intangible digital assets like crypto, creating ambiguity regarding recognition, measurement, and valuation in a Shariah-compliant manner."
            }}
        ],
        "affected_clauses": ["FAS 4 - Clause 7", "FAS 4 - Clause 9"],
        "user_context": "...",
        "FAS_reference": "FAS 4"
        }}

        Knowledge base context:
        {knowledge}

        User context:
        {context}

        FAS to review:
        {FAS}

        Respond ONLY with the JSON object.
        """
    )

def fcia_agent(fcia_input: FCIAInput) -> FCIAOutput:
    fas_namespace = get_fas_namespace(fcia_input.FAS)
    knowledge = retrieve_knowledge_from_pinecone(fcia_input.context, fas_namespace)
    prompt = build_fcia_prompt(fcia_input.context, fcia_input.FAS, knowledge)
    response = call_gemini_llm(prompt)
    print("response for fcia agent", response)
    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    data = json.loads(response)
    return FCIAOutput(**data) 