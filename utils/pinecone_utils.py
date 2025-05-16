import os
from pinecone import Pinecone
import openai
from typing import List, Optional

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

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_fas = pc.Index(PINECONE_INDEX_FAS_NAME)

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a text using OpenAI's embedding model.
    
    Args:
        text (str): The text to embed
        
    Returns:
        List[float]: The embedding vector
    """
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def retrieve_knowledge_from_pinecone_fas(query: str, fas_namespace: Optional[str] = None) -> str:
    """
    Retrieve relevant knowledge from Pinecone FAS index.
    
    Args:
        query (str): The query text
        fas_namespace (Optional[str]): The FAS namespace to search in
        
    Returns:
        str: Concatenated relevant text from top matches
    """
    try:
        embedding = get_embedding(query)
        # Query Pinecone for the top 2 matches in the correct namespace
        result = index_fas.query(
            vector=embedding,
            top_k=2,
            namespace=fas_namespace,
            include_metadata=True
        )
        
        if result and result.matches:
            # Concatenate the text of the top matches
            return '\n'.join([m.metadata.get('text', '') for m in result.matches])
        return '[No relevant FAS document found]'
        
    except Exception as e:
        print(f"Error retrieving knowledge from Pinecone: {str(e)}")
        return '[Error retrieving FAS document]' 