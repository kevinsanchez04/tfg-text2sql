import json
import sqlglot
from sqlglot import exp

DEV_FILE = "db/bird-1/dev.json"

def extract_structural_patterns(sql_query):
    """Extract structural info per table: operations, aggregations, predicates."""
    try:
        parsed = sqlglot.parse_one(sql_query)
    except Exception:
        return {}

    table_patterns, alias_map, base_tables = {}, {}, []
    subquery_aliases = {sq.alias for sq in parsed.find_all(exp.Subquery) if sq.alias}

    # Map tables and aliases
    for t in parsed.find_all(exp.Table):
        if t.name and t.name not in subquery_aliases and t.name not in base_tables:
            base_tables.append(t.name)
        if t.alias and t.alias not in subquery_aliases:
            alias_map[t.alias] = t.name

    def get_table(col):
        if col.table:
            return alias_map.get(col.table, col.table)
        return base_tables[0] if len(base_tables) == 1 else None

    def ensure(t):
        if not t or t in subquery_aliases:
            return False
        if t not in table_patterns:
            table_patterns[t] = {
                "operations": {"WHERE":0,"GROUP BY":0,"ORDER BY":0,"LIMIT":0},
                "aggregations": {"COUNT":0,"DISTINCT":0,"MAX":0,"MIN":0,"AVG":0},
                "predicates": {}
            }
        return True

    # WHERE
    implicit_join_nodes = set()
    for w in parsed.find_all(exp.Where):
        for eq in w.find_all(exp.EQ):
            if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
                t_left = get_table(eq.left)
                t_right = get_table(eq.right)

                # If the columns belong to different tables, this is an implicit join
                if t_left and t_right and t_left != t_right:
                    implicit_join_nodes.add(id(eq.left))
                    implicit_join_nodes.add(id(eq.right))

    # Count the actual filter predicates
    for w in parsed.find_all(exp.Where):
        touched = set()
        for c in w.find_all(exp.Column):
            # Skip columns that were used in an implicit join
            if id(c) in implicit_join_nodes:
                continue

            t = get_table(c)
            if t and ensure(t):
                touched.add(t)
                name = c.name.lower()
                preds = table_patterns[t]["predicates"]
                preds[name] = preds.get(name, 0) + 1

        # Count the WHERE operation only if the table has actual filter predicates
        for t in touched:
            table_patterns[t]["operations"]["WHERE"] += 1

    # GROUP BY
    for g in parsed.find_all(exp.Group):
        touched = {get_table(c) for c in g.find_all(exp.Column) if ensure(get_table(c))}
        for t in touched:
            table_patterns[t]["operations"]["GROUP BY"] += 1

    # ORDER BY
    for o in parsed.find_all(exp.Order):
        touched = {get_table(c) for c in o.find_all(exp.Column) if ensure(get_table(c))}
        for t in touched:
            table_patterns[t]["operations"]["ORDER BY"] += 1

    # LIMIT
    if parsed.find(exp.Limit):
        touched = {get_table(c) for c in parsed.find_all(exp.Column) if ensure(get_table(c))}
        for t in touched:
            table_patterns[t]["operations"]["LIMIT"] += 1

    # Aggregations
    for name, node in [("COUNT",exp.Count),("MAX",exp.Max),("MIN",exp.Min),("AVG",exp.Avg)]:
        for agg in parsed.find_all(node):
            touched = {get_table(c) for c in agg.find_all(exp.Column) if ensure(get_table(c))}
            for t in touched:
                table_patterns[t]["aggregations"][name] += 1

    # DISTINCT
    for d in parsed.find_all(exp.Distinct):
        touched = {get_table(c) for c in d.find_all(exp.Column) if ensure(get_table(c))}
        for t in touched:
            table_patterns[t]["aggregations"]["DISTINCT"] += 1

    return table_patterns


def ast_parser():
    queries = json.load(open(DEV_FILE))
    toxicology_queries = [q for q in queries if 195 <= q["question_id"] <= 339]

    all_asts = []

    for q in toxicology_queries:
        query_id = q["question_id"]
        sql_query = q["SQL"]

        nodes = []
        edges = []
        columns = []
        alias_map = {}
        source_table = None

        try:
            sql_root = sqlglot.parse_one(sql_query)

            subquery_aliases = {subq.alias for subq in sql_root.find_all(exp.Subquery) if subq.alias}

            # Extract nodes and alias map
            for t in sql_root.find_all(exp.Table):
                if t.name and t.name not in nodes and t.name not in subquery_aliases:
                    nodes.append(t.name)
                if t.alias and t.alias not in subquery_aliases:
                    alias_map[t.alias] = t.name

            # Extract join edges
            for j in sql_root.find_all(exp.Join):
                on_clause = j.args.get("on")
                if on_clause:
                    
                    join_cols = list(on_clause.find_all(exp.Column))
                    tables_in_join = set()
                    for c in join_cols:
                        if c.table and c.table not in subquery_aliases:
                            tables_in_join.add(alias_map.get(c.table, c.table))
                    
                    tables_in_join = list(tables_in_join)
                    if len(tables_in_join) == 2:
                        condition_str = on_clause.sql()

                        # Replace alias
                        for alias, real_name in alias_map.items():
                            condition_str = condition_str.replace(f"{alias}.", f"{real_name}.")
                        
                        edges.append({
                            "source": tables_in_join[0],
                            "target": tables_in_join[1],
                            "condition": condition_str
                        })
            
            # Extract implicit join edges from the WHERE clause
            for w in sql_root.find_all(exp.Where):
                for eq in w.find_all(exp.EQ):
                    left = eq.left
                    right = eq.right

                    # Check that both operands are columns
                    if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                        table_left = left.table
                        table_right = right.table

                        # Skip columns that belong to subquery aliases
                        if table_left and table_right and table_left not in subquery_aliases and table_right not in subquery_aliases:
                            real_table_left = alias_map.get(table_left, table_left)
                            real_table_right = alias_map.get(table_right, table_right)
                            
                            # If the tables are different, this is an implicit join
                            if real_table_left != real_table_right:
                                col_left_full = f"{real_table_left}.{left.name}"
                                col_right_full = f"{real_table_right}.{right.name}"
                                
                                edges.append({
                                    "source": real_table_left,
                                    "target": real_table_right,
                                    "condition": f"{col_left_full} = {col_right_full}"
                                })


            # Extract columns
            for c in sql_root.find_all(exp.Column):
                col_name = c.name
                table_name = c.table

                if table_name in subquery_aliases:
                    continue

                if table_name:
                    real_table = alias_map.get(table_name, table_name)
                    full_col = f"{real_table}.{col_name}"
                else:
                    full_col = col_name

                if full_col not in columns:
                    columns.append(full_col)

            # Structural patterns (full AST)
            table_patterns = extract_structural_patterns(sql_query)

            all_asts.append({
                "query_id": query_id,
                "original_query": sql_query,
                "nodes": nodes,
                "edges": edges,
                "columns": columns,
                "structural_patterns": table_patterns
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
    with open("data/ast/ast.json", "w") as f:
        json.dump(results, f, indent=4)
