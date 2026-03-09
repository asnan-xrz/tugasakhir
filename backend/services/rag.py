# Stub for RAG/ChromaDB queries

async def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieves relevant text segments from ChromaDB based on a query.
    Used to ground the scene generation within the context of the uploaded script.
    """
    print(f"Retrieving context for query: {query}")
    # Return fake context
    return "The scene takes place in a dimly lit cyberpunk alleyway. Rain is pouring down."
