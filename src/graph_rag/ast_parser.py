import json
import sqlglot
from sqlglot import exp

DEV_FILE = "db/bird-1/dev.json"

def ast_parser():
    queries = json.load(open(DEV_FILE))
    # Filter for toxicology dataset
    toxicology_queries = [q for q in queries if 195 <= q["question_id"] <= 339]
    
    all_asts = []

    for q in toxicology_queries:
        query_id = q["question_id"]
        sql_query = q["SQL"]
        
        nodes = []
        edges = []
        extra_logic = []
        columns = []
        alias_map = {}
        source_table = None

        try:
            sql_root = sqlglot.parse_one(sql_query)
            
            # Extract nodes and build alias map
            for t in sql_root.find_all(exp.Table):
                if t.name and t.name not in nodes:
                    nodes.append(t.name)
                if t.alias:
                    alias_map[t.alias] = t.name

            # Extract origin (FROM)
            for f in sql_root.find_all(exp.From):
                table_node = f.find(exp.Table)
                if table_node:
                    source_table = table_node.name

            # Extract edges (JOINS) and resolve aliases in conditions
            for j in sql_root.find_all(exp.Join):
                target_table = j.find(exp.Table).name
                
                on_clause = j.args.get("on")
                condition = on_clause.sql() if on_clause else ""
                
                # Replace aliases with real table names in the condition string
                for alias, table_name in alias_map.items():
                    condition = condition.replace(f"{alias}.", f"{table_name}.")
                    
                edges.append({
                    "source": source_table,
                    "target": target_table,
                    "condition": condition
                })
            
            # Columns
            for c in sql_root.find_all(exp.Column):
                col_name = c.name
                table_name = c.table

                if table_name:
                    real_table = alias_map.get(table_name, table_name)
                    full_col = f"{real_table}.{col_name}"
                else:
                    full_col = col_name
                if full_col and full_col not in columns:
                    columns.append(full_col)    

            # Extra logic (WHERE, GROUP BY, ORDER BY)
            for clause_type in [exp.Where, exp.Group, exp.Order]:
                clause_node = sql_root.find(clause_type)
                if clause_node:
                    clause_str = clause_node.sql()
                    for alias, table_name in alias_map.items():
                        clause_str = clause_str.replace(f"{alias}.", f"{table_name}.")
                    extra_logic.append(clause_str)
            
            # Append parsed query data
            all_asts.append({
                "query_id": query_id,
                "original_query": sql_query,
                "nodes": nodes, 
                "edges": edges,
                "columns": columns,
                "extra_logic": extra_logic
            })
            
        except Exception as e:
            print(f"Skipping query {query_id} due to parse error: {e}")

    print(f"\nPARSED: {len(all_asts)} ASTs")
    if all_asts:
        print("Example (Query 195):")
        print(json.dumps(all_asts[0], indent=4))
        
    return all_asts

if __name__ == "__main__":
    results = ast_parser()
    with open("test_ast.json", "w") as f:
        json.dump(results, f, indent=4)