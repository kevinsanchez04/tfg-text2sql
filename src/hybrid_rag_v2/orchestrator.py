# src/hybrid_rag_v2/orchestrator.py
from vector_rag.retriever import get_similar_queries_context
from hybrid_rag_v2.graph_navigator import graph_navigator
from graph_rag.schema_linker import extract_tables_from_query


VALUE_LINKING_GUARD = (
    "\n### VALUE LINKING INSTRUCTIONS ###\n"
    "1. For literal string/symbol values in WHERE clauses (codes, abbreviations, "
    "symbols like '=', '-', '#', or single/double-letter element codes), you MUST "
    "trust the 'CRITICAL - EXACT VALUES' entries in the SCHEMA MAP and the literal "
    "values used in the 'Expected SQL' of the REFERENCE EXAMPLES below, over your own "
    "general knowledge of how such concepts are usually spelled out "
    "(e.g. write 'i', not 'Iodine'; write '=', not 'Double').\n"
    "2. For table selection and JOIN structure, follow ONLY the SCHEMA MAP and "
    "STRUCTURAL HINT above. Do NOT add a table just because a REFERENCE EXAMPLE below "
    "happens to use it, if that table is not required to answer the current question.\n"
)


def get_hybrid_context(nl_query: str, query_id: int, llm, spark_sql) -> str:
    """
    Combines structural context from the pruned Graph (table/JOIN selection —
    authoritative) with semantic context from the Vector DB (value/vocabulary
    linking — authoritative for literals only).
    """

    # --- Structural context (pruned schema + join hints) ---
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

    # --- Semantic context (few-shot examples, for value/vocabulary linking) ---
    vector_context = get_similar_queries_context(nl_query, exclude_query_id=query_id)

    final_context = (
        f"{graph_context}"
        f"{hint_text}"
        f"\n### REFERENCE EXAMPLES (similar past NL->SQL pairs) ###\n"
        f"{vector_context}"
        f"{VALUE_LINKING_GUARD}"
    )

    return final_context