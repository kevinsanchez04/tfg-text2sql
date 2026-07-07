# src/hybrid_rag/orchestrator.py
from vector_rag.retriever import get_similar_queries_context
from graph_rag.graph_navigator import graph_navigator
from graph_rag.schema_linker import extract_tables_from_query

def get_hybrid_context(nl_query: str, query_id: int, llm, spark_sql) -> str:
    """
    Combines semantic context from Vector DB and structural context from Graph.
    """
    
    # Get semantic context (Few-Shot examples)
    vector_context = get_similar_queries_context(nl_query, exclude_query_id=query_id)
    
    # Get structural context (Graph joins and metadata)
    all_tables = spark_sql.get_usable_table_names()
    tables = extract_tables_from_query(nl_query, all_tables, llm)
    graph_context = graph_navigator(tables)
    
    # Combine both into a single enriched prompt block
    enriched_context = f"""
        --- SEMANTIC CONTEXT (Examples) ---
        {vector_context}

        --- STRUCTURAL CONTEXT (Graph Rules) ---
        {graph_context}
        """
    return enriched_context