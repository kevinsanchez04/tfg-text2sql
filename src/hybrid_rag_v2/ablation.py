# src/hybrid_rag_v2/ablation.py
"""Variants d'ablació interna (leave-one-out) de Hybrid RAG V2.

Aquest mòdul NO reimplementa cap mecanisme de recuperació: reutilitza
directament vector_rag.retriever, graph_rag.schema_linker i els dos
navegadors de graf ja existents (hybrid_rag_v2.graph_navigator, ponderat amb
ALPHA, i graph_rag.graph_navigator, purament freqüencial, el mateix que fa
servir Hybrid V1). Cada funció d'aquest fitxer desactiva EXACTAMENT un dels
tres components de Hybrid V2 (resolució d'entitats, navegació ponderada
ALPHA, negative prompting), mantenint els altres dos intactes, tal com
s'especifica a la Secció 2.2.3 de la memòria.
"""

from vector_rag.retriever import get_similar_queries_context
from graph_rag.schema_linker import extract_tables_from_query
from hybrid_rag_v2.graph_navigator import graph_navigator as graph_navigator_weighted
from graph_rag.graph_navigator import graph_navigator as graph_navigator_unweighted


# Bloc d'instruccions complet, idèntic al de hybrid_rag_v2.orchestrator.INSTRUCTIONS.
INSTRUCTIONS_FULL = """--- CRITICAL INSTRUCTIONS ---
You are a master SparkSQL data analyst. 
1. Use the Semantic Examples and Historical Metadata ONLY as a structural reference for the SparkSQL dialect and schema logic.
2. DO NOT blindly copy the examples and DO NOT hardcode values from the Historical Metadata unless explicitly requested by the user. If the question asks for aggregates ("most common", "top", "maximum"), you MUST use dynamic SQL (GROUP BY, ORDER BY DESC, LIMIT 1) to calculate it, NEVER hardcode the WHERE clause to guess the answer.
3. Follow the Structural Context strictly for JOINs and table routing.
4. STRICT CASING: Pay close attention to the capitalization in the Sample Data. If the sample values for a column are strictly lowercase, you MUST convert your filtering strings to lowercase to match the database format.
"""

# Bloc reduït per a l'ablació de negative prompting: es manté únicament la
# instrucció estructural (punt 3 de l'original). Es retiren la instrucció
# anti-hardcoding/anti-al·lucinació (punt 2) i la regla de capitalització
# estricta (punt 4), que constitueixen conjuntament el mecanisme de negative
# prompting objecte d'estudi.
INSTRUCTIONS_NO_NEGATIVE_PROMPTING = """--- INSTRUCTIONS ---
You are a master SparkSQL data analyst.
1. Use the Semantic Examples and Historical Metadata as a structural reference for the SparkSQL dialect and schema logic.
2. Follow the Structural Context strictly for JOINs and table routing.
"""


def _shared_retrieval(nl_query, query_id, llm, spark_sql, db_name):
    """Pas comú a les tres variants: recuperació vectorial (LOO) i selecció
    de taules + mostres d'entitat via el schema linker existent."""
    vector_result = get_similar_queries_context(
        nl_query, exclude_query_id=query_id, db_name=db_name
    )
    vector_text = vector_result[0] if isinstance(vector_result, tuple) else vector_result

    all_tables = spark_sql.get_usable_table_names()
    selected_tables, entity_samples = extract_tables_from_query(
        nl_query, all_tables, llm, db_name=db_name
    )

    return vector_text, selected_tables, entity_samples


def _append_entity_block(enriched_context, entity_samples):
    if entity_samples:
        enriched_context += "--- ENTITY RESOLUTION (Sample Values) ---\n"
        for table, cols in entity_samples.items():
            enriched_context += f"Table [{table}]:\n"
            for col, vals in cols.items():
                enriched_context += f"  - Column '{col}' matches categorical values: {vals}\n"
        enriched_context += "\n"
    return enriched_context


def get_hybrid_context_no_entity_resolution(nl_query, query_id, llm, spark_sql, embedder, db_name):
    """Hybrid V2 sense el bloc ENTITY RESOLUTION (es manté ALPHA + negative prompting)."""
    vector_text, selected_tables, _entity_samples = _shared_retrieval(
        nl_query, query_id, llm, spark_sql, db_name
    )

    graph_context = graph_navigator_weighted(
        nl_query=nl_query, tables=selected_tables, embedder=embedder, db_name=db_name
    )

    enriched_context = INSTRUCTIONS_FULL
    enriched_context += f"\n\n--- SEMANTIC CONTEXT (Examples) ---\n{vector_text}\n\n"
    # Bloc ENTITY RESOLUTION intencionadament omès.
    enriched_context += f"--- STRUCTURAL CONTEXT (Graph Rules) ---\n{graph_context}\n\n"
    enriched_context += f"--- USER QUESTION ---\nGenerate the SparkSQL query to answer the following question:\n{nl_query}\n\n"

    return enriched_context


def get_hybrid_context_no_alpha_weighting(nl_query, query_id, llm, spark_sql, embedder, db_name):
    """Hybrid V2 amb el navegador de graf revertit al càlcul purament
    freqüencial (equivalent al de Hybrid V1), sense el terme semàntic
    ponderat. Es manté ENTITY RESOLUTION i negative prompting."""
    vector_text, selected_tables, entity_samples = _shared_retrieval(
        nl_query, query_id, llm, spark_sql, db_name
    )

    # Navegador no ponderat: no intervé embedder ni cap terme semàntic.
    graph_context = graph_navigator_unweighted(selected_tables, db_name)

    enriched_context = INSTRUCTIONS_FULL
    enriched_context += f"\n\n--- SEMANTIC CONTEXT (Examples) ---\n{vector_text}\n\n"
    enriched_context = _append_entity_block(enriched_context, entity_samples)
    enriched_context += f"--- STRUCTURAL CONTEXT (Graph Rules) ---\n{graph_context}\n\n"
    enriched_context += f"--- USER QUESTION ---\nGenerate the SparkSQL query to answer the following question:\n{nl_query}\n\n"

    return enriched_context


def get_hybrid_context_no_negative_prompting(nl_query, query_id, llm, spark_sql, embedder, db_name):
    """Hybrid V2 amb el bloc d'instruccions crítiques reduït (sense negative
    prompting). Es manté ENTITY RESOLUTION i la navegació ponderada ALPHA."""
    vector_text, selected_tables, entity_samples = _shared_retrieval(
        nl_query, query_id, llm, spark_sql, db_name
    )

    graph_context = graph_navigator_weighted(
        nl_query=nl_query, tables=selected_tables, embedder=embedder, db_name=db_name
    )

    enriched_context = INSTRUCTIONS_NO_NEGATIVE_PROMPTING
    enriched_context += f"\n\n--- SEMANTIC CONTEXT (Examples) ---\n{vector_text}\n\n"
    enriched_context = _append_entity_block(enriched_context, entity_samples)
    enriched_context += f"--- STRUCTURAL CONTEXT (Graph Rules) ---\n{graph_context}\n\n"
    enriched_context += f"--- USER QUESTION ---\nGenerate the SparkSQL query to answer the following question:\n{nl_query}\n\n"

    return enriched_context