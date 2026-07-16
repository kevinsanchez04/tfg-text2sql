import json
import os
from paths import get_graph_path

def load_schema_metadata_from_graph(db_name):
    """
    Load schema information from the property graph and build a dictionary
    containing columns and raw sample values for each table.
    """
    graph_path = get_graph_path(db_name)
    if not os.path.exists(graph_path):
        return {}
    with open(graph_path, "r") as file:
        graph_data = json.load(file)

    schema_metadata = {}

    for table_node in graph_data.get("nodes", []):
        table_name = table_node.get("id")
        sample_values = table_node.get("sample_values", {})
        column_frequencies = table_node.get("columns_freq", {})

        # Store raw lists instead of formatted strings to allow filtering later
        schema_metadata[table_name] = {
            "columns": list(column_frequencies.keys()),
            "samples": sample_values
        }

    return schema_metadata


def extract_tables_from_query(nl_query: str, available_tables: list, llm, db_name: str) -> tuple:
    """
    Use the language model to determine which tables are needed to answer
    the user's question, and strictly filter sample values to prevent data leakage.
    
    Returns:
        tuple: (selected_tables: list, filtered_sample_values: dict)
    """
    schema_metadata = load_schema_metadata_from_graph(db_name)

    # Build context for the router LLM
    router_context = {}
    for table, meta in schema_metadata.items():
        if table in available_tables:
            router_context[table] = {}
            for col in meta["columns"]:
                if col in meta["samples"]:
                    router_context[table][col] = f"Examples: {meta['samples'][col]}"
                else:
                    router_context[table][col] = "No example values available"

    if router_context:
        schema_context = json.dumps(router_context, indent=2)
    else:
        schema_context = str(available_tables)

    prompt = f"""You are a strict database routing assistant.

    Below is the database schema with table names, their columns, and example values:

    {schema_context}

    User question:
    "{nl_query}"

    Determine which tables are strictly required to answer the question.
    CRITICAL RULE: Be extremely minimal. Only select tables that contain columns explicitly needed for the SELECT or WHERE clauses.
    
    Return ONLY a valid JSON array containing the exact table names.

    Example:
    ["table_a", "table_b"]
    """

    selected_tables = []
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        
        parsed_tables = json.loads(response_text)
        selected_tables = [t for t in parsed_tables if t in available_tables]

    except Exception as error:
        print(f"[Schema Router] Failed to extract tables: {error}")
        return [], {}

    filtered_sample_values = {}
    
    for table in selected_tables:
        table_samples = schema_metadata.get(table, {}).get("samples", {})
        if table_samples:
            filtered_sample_values[table] = table_samples

    return selected_tables, filtered_sample_values