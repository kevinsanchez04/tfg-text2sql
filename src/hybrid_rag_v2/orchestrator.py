# src/hybrid_rag/orchestrator.py
from vector_rag.retriever import get_similar_queries_context
from hybrid_rag_v2.graph_navigator import graph_navigator
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
    graph_context, best_paths = graph_navigator(tables, nl_query, llm)
    
    hint_text = ""
    if best_paths:
        chains = [" -> ".join(p) for p in best_paths.values()]
        hint_text = (
            "\n### STRUCTURAL HINT (guidance only, not a hard rule) ###\n"
            + "\n".join(f"- Suggested path: {c}" for c in chains)
            + "\nUse this only if it matches the question's logic. If the question can be "
            "answered with fewer tables than shown here, prefer the simpler query.\n"
        )

    final_graph_context = graph_context + hint_text
    return final_graph_context