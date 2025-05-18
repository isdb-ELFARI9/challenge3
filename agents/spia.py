from schemas.spia import SPIAInput, SPIAOutput
from utils.llm import call_gemini_llm
import os
import json
from pinecone import Pinecone
import openai

from utils.write_to_file import write_to_file

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
        f"""You are the Shariah Principles Integration Agent (SPIA) for an Islamic finance standards review system.

        Role: You are a senior Shariah board scholar and standards developer with deep expertise in applying classical and contemporary Shariah principles to modern financial transactions and instruments.

        Your task is to:
        1. Carefully review the gap report, user context (including its potential implications for Islamic finance), and the relevant FAS.
        2. Consult the provided Shariah standards knowledge base for relevant principles, rules, fatwas, and scholarly opinions specifically addressing the type of activities or instruments implied by the user context and identified in the gap report (e.g., digital assets like crypto, specific contract types, etc.).
        3. Propose a clear, detailed, and Shariah-compliant solution or set of principles to fill the identified gap. Your solution must be robust enough to guide IFIs engaging in the potential activities highlighted by the user context.
        4. For each new or revised Shariah guidance clause derived from your solution, provide:
            - "clause_id": a unique identifier (e.g., 'FAS[X].SH[Y]')
            - "text": the full, precisely worded clause suitable for inclusion in a standard
            - "reference": cite the specific Shariah standard, fatwa number, or scholarly consensus from the knowledge base or general Shariah principles supporting this clause
        5. Provide a comprehensive justification for your overall Shariah solution and for each proposed clause. Explain *how* it aligns with fundamental Shariah principles, addresses the specific gap identified by FCIA, and accommodates the potential activities related to the user context (e.g., "This clause on crypto ownership aligns with the principle of 'mal mutaqawwim' (valuable asset) as defined in [Reference], addressing the gap in asset definition for digital tokens identified by FCIA which is crucial for potential crypto trading activities").
        6. List all specific references used in the "references" array.

        Output your findings in the following JSON format:
        {{
        "shariah_solution": "...",
        "updated_shariah_clauses": [
            {{
            "clause_id": "FAS4.SH1",
            "text": "...",
            "reference": "AAOIFI Shariah Standard 21"
            }}
        ],
        "references": ["AAOIFI Shariah Standard 21", "Specific Fatwa"]
        }}

        Knowledge base context:
        {knowledge}

        Gap report:
        {gap_report}

        User context:
        {user_context}

        FAS to review:
        {FAS}

        Respond ONLY with the JSON object.
        """
    )

def spia_agent(spia_input: SPIAInput, llm_name="gemini") -> SPIAOutput:
    # For this agent, always use the SS index and the namespace should be based on the most relevant SS (simulate with first in knowledge_indexes)
    ss_namespace = spia_input.knowledge_indexes[0] if spia_input.knowledge_indexes else None
    knowledge = retrieve_knowledge_from_pinecone_ss(str(spia_input.gap_report), ss_namespace)
    prompt = build_spia_prompt(spia_input.gap_report, spia_input.FAS, spia_input.user_context, knowledge)
    write_to_file("prompts.txt", prompt)
    if llm_name == "gemini":
        response=call_gemini_llm(prompt)
    elif llm_name == "deepseek":
        response=call_gemini_llm(prompt)
    else:
        response=call_gemini_llm(prompt)

    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    data = json.loads(response)
    return SPIAOutput(**data) 