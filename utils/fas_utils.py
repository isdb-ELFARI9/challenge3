def get_fas_namespace(fas: str) -> str:
    """
    Get the Pinecone namespace for a given FAS.
    
    Args:
        fas (str): The FAS identifier (e.g., "FAS 1")
        
    Returns:
        str: The namespace for the FAS
    """
    return f"fas_{fas.lower().replace(' ', '_')}" 