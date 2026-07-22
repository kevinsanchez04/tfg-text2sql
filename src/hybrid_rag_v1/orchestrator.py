# src/hybrid_rag_v1/orchestrator.py
from vector_rag.retriever import get_similar_queries_context
from graph_rag.graph_navigator import graph_navigator
from graph_rag.schema_linker import extract_tables_from_query

def get_hybrid_context(nl_query: str, query_id: int, llm, spark_sql, db_name: str) -> str:
    """
    Combines semantic context from Vector DB and structural context from Graph.
    """
    
    # Get semantic context (Few-Shot examples)
    vector_result = get_similar_queries_context(nl_query, exclude_query_id=query_id, db_name=db_name)
    # get_similar_queries_context returns (text, distance)
    vector_context = vector_result[0] if isinstance(vector_result, tuple) else vector_result

    # Get structural context (Graph joins and metadata)
    all_tables = spark_sql.get_usable_table_names()
    # extract_tables_from_query returns (tables, samples)
    tables, _entity_samples = extract_tables_from_query(nl_query, all_tables, llm, db_name=db_name)
    graph_context = graph_navigator(tables, db_name=db_name)
    
    # Combine both into a single enriched prompt block
    enriched_context = f"""
        --- SEMANTIC CONTEXT (Examples) ---
        {vector_context}

        --- STRUCTURAL CONTEXT (Graph Rules) ---
        {graph_context}
        """
    return enriched_context