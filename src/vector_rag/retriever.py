# src/vector_rag/retriever.py
import chromadb

def get_similar_queries_context(nl_query: str, exclude_query_id: int, top_k: int = 3) -> str:
    """Look it up in Chroma DB and returns the context in string format"""
    
    # Connect Chroma
    chroma_client = chromadb.PersistentClient(path="./db/vector_store")
    collection = chroma_client.get_collection(name="toxicology_queries")
    
    # Safe top_k calculation
    total_docs = collection.count()
    safe_top_k = min(top_k, total_docs - 1)
    
    if safe_top_k <= 0:
        return "As a strict reference, here are some examples:\n"
    
    # Searching with Leave-One-Out
    results = collection.query(
        query_texts=[nl_query],
        n_results=safe_top_k,
        where={"query_id": {"$ne": exclude_query_id}} 
    )
    
    rag_context = "As a strict reference, here are some examples:\n"
    for i in range(len(results['documents'][0])):
        example_q = results['documents'][0][i]
        example_sql = results['metadatas'][0][i]['golden_sql']
        rag_context += f"- Example Question: '{example_q}'\n  Expected SQL: {example_sql}\n\n"
        
    return rag_context