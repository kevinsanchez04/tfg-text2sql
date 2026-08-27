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


# Available context generation for ablation study

def hybrid_v2_no_entity_context(nl_query, query_id, llm, spark_sql, db_name, embedder):
    from hybrid_rag_v2.ablation import get_hybrid_context_no_entity_resolution
    return get_hybrid_context_no_entity_resolution(
        nl_query, query_id, llm, spark_sql, embedder, db_name
    )


def hybrid_v2_no_alpha_context(nl_query, query_id, llm, spark_sql, db_name, embedder):
    from hybrid_rag_v2.ablation import get_hybrid_context_no_alpha_weighting
    return get_hybrid_context_no_alpha_weighting(
        nl_query, query_id, llm, spark_sql, embedder, db_name
    )


def hybrid_v2_no_negprompt_context(nl_query, query_id, llm, spark_sql, db_name, embedder):
    from hybrid_rag_v2.ablation import get_hybrid_context_no_negative_prompting
    return get_hybrid_context_no_negative_prompting(
        nl_query, query_id, llm, spark_sql, embedder, db_name
    )


CONTEXT_STRATEGIES["hybrid_v2_no_entity"] = hybrid_v2_no_entity_context
CONTEXT_STRATEGIES["hybrid_v2_no_alpha"] = hybrid_v2_no_alpha_context
CONTEXT_STRATEGIES["hybrid_v2_no_negprompt"] = hybrid_v2_no_negprompt_context


def make_hybrid_v2_alpha_context(alpha):
    """Factory: returns a context_fn identical to hybrid_v2_context but using
    the ALPHA-parametrized navigator, for the sensitivity sweep (Section 3.6)."""
    def _context_fn(nl_query, query_id, llm, spark_sql, db_name, embedder):
        from vector_rag.retriever import get_similar_queries_context
        from graph_rag.schema_linker import extract_tables_from_query
        from hybrid_rag_v2.graph_navigator import graph_navigator
        from hybrid_rag_v2.orchestrator import INSTRUCTIONS

        vector_result = get_similar_queries_context(nl_query, exclude_query_id=query_id, db_name=db_name)
        vector_text = vector_result[0] if isinstance(vector_result, tuple) else vector_result

        all_tables = spark_sql.get_usable_table_names()
        selected_tables, entity_samples = extract_tables_from_query(nl_query, all_tables, llm, db_name=db_name)
        graph_context = graph_navigator(nl_query, selected_tables, db_name, embedder=embedder, alpha=alpha)

        enriched = INSTRUCTIONS
        enriched += f"\n\n--- SEMANTIC CONTEXT (Examples) ---\n{vector_text}\n\n"
        if entity_samples:
            enriched += "--- ENTITY RESOLUTION (Sample Values) ---\n"
            for table, cols in entity_samples.items():
                enriched += f"Table [{table}]:\n"
                for col, vals in cols.items():
                    enriched += f"  - Column '{col}' matches categorical values: {vals}\n"
            enriched += "\n"
        enriched += f"--- STRUCTURAL CONTEXT (Graph Rules) ---\n{graph_context}\n\n"
        enriched += f"--- USER QUESTION ---\nGenerate the SparkSQL query to answer the following question:\n{nl_query}\n\n"
        return enriched
    return _context_fn


ALPHA_SWEEP_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for _a in ALPHA_SWEEP_VALUES:
    CONTEXT_STRATEGIES[f"hybrid_v2_alpha_{_a}"] = make_hybrid_v2_alpha_context(_a)