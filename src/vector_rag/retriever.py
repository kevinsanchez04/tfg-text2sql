# src/vector_rag/retriever.py
import chromadb
from paths import get_vector_store_path, get_vector_collection_name

def get_similar_queries_context(nl_query: str, exclude_query_id: int, db_name: str, top_k: int = 3) -> str:
    """Look it up in Chroma DB and returns the context in string format"""
    
    # Connect Chroma
    chroma_client = chromadb.PersistentClient(path=get_vector_store_path(db_name))
    collection = chroma_client.get_collection(name=get_vector_collection_name(db_name))
    
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
    
    if 'distances' in results and len(results['distances'][0]) > 0:
        best_distance = results['distances'][0][0]
    else:
        best_distance = float('inf')

    return rag_context, best_distance