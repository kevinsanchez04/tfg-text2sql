"""Adapters that unify the interface of each RAG architecture."""

def baseline_context(nl_query, query_id, llm, spark_sql, db_name, embedder=None):
    # No retrieval, return the user query.
    return nl_query


def vector_context(nl_query, query_id, llm, spark_sql, db_name, embedder=None):
    from vector_rag.retriever import get_similar_queries_context

    rag_context, _distance = get_similar_queries_context(
        nl_query, query_id, db_name
    )

    return (
        f"{rag_context}"
        f"Use the examples above to solve the following question:\n"
        f"Current Question: {nl_query}"
    )


def graph_context(nl_query, query_id, llm, spark_sql, db_name, embedder=None):
    from graph_rag.graph_navigator import graph_navigator
    from graph_rag.schema_linker import extract_tables_from_query

    all_tables = spark_sql.get_usable_table_names()
    tables_needed, _samples = extract_tables_from_query(
        nl_query, all_tables, llm, db_name
    )

    graph_ctx = graph_navigator(tables_needed, db_name)

    return (
        f"Historical Context from Graph RAG:\n"
        f"{graph_ctx}\n\n"
        f"Use this context to solve the following question:\n"
        f"Current Question: {nl_query}"
    )


def hybrid_v1_context(nl_query, query_id, llm, spark_sql, db_name, embedder=None):
    from hybrid_rag_v1.orchestrator import get_hybrid_context

    hybrid_ctx = get_hybrid_context(
        nl_query, query_id, llm, spark_sql, db_name
    )

    return (
        f"{hybrid_ctx}\n\n"
        f"Use this context to solve the following question:\n"
        f"Current Question: {nl_query}"
    )


def hybrid_v2_context(nl_query, query_id, llm, spark_sql, db_name, embedder):
    from hybrid_rag_v2.orchestrator import get_hybrid_context

    # Build context with Hybrid RAG v2.
    return get_hybrid_context(
        nl_query, query_id, llm, spark_sql, embedder, db_name
    )


# Available context generation strategies.
CONTEXT_STRATEGIES = {
    "baseline": baseline_context,
    "vector": vector_context,
    "graph": graph_context,
    "hybrid_v1": hybrid_v1_context,
    "hybrid_v2": hybrid_v2_context,
}