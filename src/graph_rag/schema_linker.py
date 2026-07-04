import json

def extract_tables_from_query(nl_query: str, available_tables: list, llm) -> list:
    """
    Extracts required tables for a given natural language query using the provided LLM.
    Returns a validated list of table names.
    """
    prompt = f"""You are a strict database routing assistant.
Available tables: {available_tables}
User question: "{nl_query}"

Which tables are strictly required to answer this question?
Return ONLY a valid JSON array of strings containing the exact table names. 
Do not include markdown formatting, explanations, or extra text.
Example: ["table_a", "table_b"]"""

    try:
        response = llm.invoke(prompt)
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        # clean potential markdown blocks injected by the llm
        clean_content = content.strip().replace("```json", "").replace("```", "").strip()
        extracted_tables = json.loads(clean_content)
        
        # filter out hallucinations
        valid_tables = [t for t in extracted_tables if t in available_tables]
        return valid_tables
        
    except Exception as e:
        print(f"[Schema Linker] Extraction failed: {str(e)}")
        return []