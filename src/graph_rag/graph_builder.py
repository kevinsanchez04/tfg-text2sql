import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import json

def build_graph(spark=None, filename='data/ast/ast.json'):
    with open(filename) as json_file:
        ast = json.load(json_file)

    G = nx.Graph()

    for query in ast:
        query_nodes = query.get("nodes", [])
        
        # Process nodes and update frequency metadata for the Property Graph
        for node in query_nodes:
            if not G.has_node(node):
                # Initialize node with empty property dictionaries
                G.add_node(node, columns_freq={}, logic_freq={}, sample_values={})
            
            # Extract and count column usage statistics
            for col in query.get("columns", []):
                if "." in col:
                    # Handle explicit references (e.g., "table.column")
                    prefix, col_name = col.split(".", 1)
                    if prefix == node:
                        current = G.nodes[node]["columns_freq"].get(col_name, 0)
                        G.nodes[node]["columns_freq"][col_name] = current + 1
                else:
                    # Handle implicit references only if the query has a single table to avoid data poisoning
                    if len(query_nodes) == 1:
                        current = G.nodes[node]["columns_freq"].get(col, 0)
                        G.nodes[node]["columns_freq"][col] = current + 1
                   
            # Track how often specific logic (WHERE, GROUP BY, etc.) is applied to this node
            for logic in query.get("extra_logic", []):
                if node in logic:
                    current_freq = G.nodes[node]["logic_freq"].get(logic, 0)
                    G.nodes[node]["logic_freq"][logic] = current_freq + 1

        # Process relationships (JOINs) between tables
        for edge in query.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            condition = edge.get("condition")

            if not source or not target:
                continue

            # Sort nodes alphabetically so A->B and B->A resolve to the same undirected edge
            sorted_nodes = sorted([source, target])
            u, v = sorted_nodes[0], sorted_nodes[1]

            # Normalize equality conditions (e.g., "A.id = B.id" becomes identical to "B.id = A.id")
            if condition and "=" in condition:
                parts = [p.strip() for p in condition.split("=")]
                normalized_condition = " = ".join(sorted(parts))
            else:
                normalized_condition = condition

            # Increment weight if the identical relationship exists, otherwise initialize it
            if G.has_edge(u, v) and G[u][v].get("condition") == normalized_condition:
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1, condition=normalized_condition)

    # --- BUILD-TIME ENTITY RESOLUTION ---
    # Fetch sample values for relevant columns and store them statically in the graph
    if spark is not None:
        print("\nStarting Build-Time Entity Resolution...")
        for node in G.nodes():
            sample_values = {}
            # Only fetch data for columns that actually appeared in the AST
            columns_to_resolve = list(G.nodes[node].get("columns_freq", {}).keys())
            
            for col in columns_to_resolve:
                try:
                    # Fetch up to 3 distinct, non-null categorical values
                    query = f"SELECT DISTINCT `{col}` FROM `{node}` WHERE `{col}` IS NOT NULL LIMIT 3"
                    rows = spark.sql(query).collect()
                    vals = [str(row[0]) for row in rows if row[0] is not None]
                    
                    if vals:
                        sample_values[col] = vals
                except Exception:
                    # Silently skip if column doesn't exist or has a complex un-stringifiable type
                    continue
                    
            G.nodes[node]["sample_values"] = sample_values
            if sample_values:
                print(f"  -> Resolved entities for [{node}]: {list(sample_values.keys())}")
    else:
        print("\nWarning: No Spark session provided. Entity Resolution skipped.")

    return G

def save_graph(G, filename='data/graphs/property_graph.json'):
    data = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nGraph successfully saved to {filename}")

def verify_graph(G):
    print(f"\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}\n")
    
    for node, data in G.nodes(data=True):
        print(f"[{node}]")
        print(f"  Cols: {data.get('columns_freq')}")
        print(f"  Logic: {data.get('logic_freq')}")
        if data.get('sample_values'):
            print(f"  Samples: {data.get('sample_values')}")
        print()

    for u, v, data in G.edges(data=True):
        print(f"[{u}] <-> [{v}] (w:{data.get('weight')})")
        print(f"  Cond: {data.get('condition')}\n")

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=9, edge_color='gray')
    edge_labels = {(u, v): f"w:{d.get('weight')}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.savefig("data/graphs/visual_graph.png", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    # Import tools specifically for running this script standalone
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
    
    try:
        from utils import ensure_sqlite_jdbc_driver
        from load_db import load_tables
        from spark_nl import get_spark_session
        
        # Initialize Spark locally for Graph Building
        jdbc_jar_path = ensure_sqlite_jdbc_driver()
        spark = get_spark_session(extra_configs={
            "spark.jars": jdbc_jar_path,
            "spark.driver.extraClassPath": jdbc_jar_path,
        })
        # Load the target database into Spark before querying
        load_tables(spark, "toxicology")
        
        G = build_graph(spark=spark)
        spark.stop()
        
    except ImportError:
        print("Could not load Spark dependencies. Building graph without Entity Resolution.")
        G = build_graph()

    save_graph(G)
    verify_graph(G)