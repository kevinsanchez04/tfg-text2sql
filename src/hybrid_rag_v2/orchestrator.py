from vector_rag.retriever import get_similar_queries_context
from hybrid_rag_v2.graph_navigator import graph_navigator
from graph_rag.schema_linker import extract_tables_from_query

INSTRUCTIONS = """--- CRITICAL INSTRUCTIONS ---
You are a master SparkSQL data analyst. 
1. Use the Semantic Examples and Historical Metadata ONLY as a structural reference for the SparkSQL dialect and schema logic.
2. DO NOT blindly copy the examples and DO NOT hardcode values from the Historical Metadata unless explicitly requested by the user. If the question asks for aggregates ("most common", "top", "maximum"), you MUST use dynamic SQL (GROUP BY, ORDER BY DESC, LIMIT 1) to calculate it, NEVER hardcode the WHERE clause to guess the answer.
3. Follow the Structural Context strictly for JOINs and table routing.
4. STRICT CASING: Pay close attention to the capitalization in the Sample Data. If the sample values for a column are strictly lowercase, you MUST convert your filtering strings to lowercase to match the database format.
"""

def get_hybrid_context(nl_query: str, query_id: int, llm, spark_sql, embedder) -> str:
    """
    Combines semantic context from Vector DB, structural context from Graph,
    and filtered entity resolution samples.
    """
    # Fetch semantic context (Few-Shot examples)
    vector_context = get_similar_queries_context(nl_query, exclude_query_id=query_id)
    vector_text = vector_context[0] if isinstance(vector_context, tuple) else vector_context

    # Fetch required tables and strictly filtered sample values
    all_tables = spark_sql.get_usable_table_names()
    selected_tables, entity_samples = extract_tables_from_query(nl_query, all_tables, llm)
    
    # Fetch structural context (AST Patterns and JOIN paths)
    graph_context = graph_navigator(nl_query=nl_query, tables=selected_tables, embedder=embedder)
    
    # Build the final prompt sequentially
    enriched_context = INSTRUCTIONS
    enriched_context += f"\n\n--- SEMANTIC CONTEXT (Examples) ---\n{vector_text}\n\n"
    
    # Conditionally inject Entity Resolution ONLY if relevant samples were found
    if entity_samples:
        enriched_context += "--- ENTITY RESOLUTION (Sample Values) ---\n"
        for table, cols in entity_samples.items():
            enriched_context += f"Table [{table}]:\n"
            for col, vals in cols.items():
                enriched_context += f"  - Column '{col}' matches categorical values: {vals}\n"
        enriched_context += "\n"

    enriched_context += f"--- STRUCTURAL CONTEXT (Graph Rules) ---\n{graph_context}\n\n"

    enriched_context += f"--- USER QUESTION ---\nGenerate the SparkSQL query to answer the following question:\n{nl_query}\n\n"
    
    
    return enriched_context