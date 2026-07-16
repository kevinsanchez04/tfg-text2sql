import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paths import get_ast_path, get_graph_path
import argparse

def build_graph(db_name, spark=None):
    filename = get_ast_path(db_name)
    with open(filename) as json_file:
        ast = json.load(json_file)

    # Directed graph to preserve join semantics
    G = nx.DiGraph()

    for query in ast:
        query_nodes = query.get("nodes", [])
        query_sql = query.get("original_query", "")

        # Filter out synthetic or invalid nodes (e.g., SubQuery)
        query_nodes = [n for n in query_nodes if n and n.lower() != "subquery"]

        if not query_nodes:
            continue

        # Create nodes with structural metadata
        for node in query_nodes:
            if not G.has_node(node):
                G.add_node(
                    node,
                    columns_freq={},
                    operations_freq={"WHERE": 0, "GROUP BY": 0, "ORDER BY": 0, "LIMIT": 0},
                    aggregations_freq={"COUNT": 0, "DISTINCT": 0, "MAX": 0, "MIN": 0, "AVG": 0},
                    predicates_freq={},
                    sample_values={}
                )

        # Column usage frequency
        for col in query.get("columns", []):
            if "." in col:
                prefix, col_name = col.split(".", 1)
                # Use prefix as table name directly (AST already resolved aliases)
                if prefix in query_nodes:
                    current = G.nodes[prefix]["columns_freq"].get(col_name, 0)
                    G.nodes[prefix]["columns_freq"][col_name] = current + 1
            else:
                if len(query_nodes) == 1:
                    node = query_nodes[0]
                    current = G.nodes[node]["columns_freq"].get(col, 0)
                    G.nodes[node]["columns_freq"][col] = current + 1

        # Structural patterns at query level
        patterns = query.get("structural_patterns", {})

        # Assign operations and aggregations to all involved tables (query-level patterns)
        for node in query_nodes:
            node_patterns = patterns.get(node, {})
            ops = node_patterns.get("operations", {})
            aggs = node_patterns.get("aggregations", {})
            preds = node_patterns.get("predicates", {})

            for k, v in ops.items():
                G.nodes[node]["operations_freq"][k] += v
            for k, v in aggs.items():
                G.nodes[node]["aggregations_freq"][k] += v

        # Assign predicate columns only to tables where those columns appear
        for pred_col, freq in preds.items():
            for node in query_nodes:
                cols = G.nodes[node].get("columns_freq", {})
                if pred_col in cols:
                    current = G.nodes[node]["predicates_freq"].get(pred_col, 0)
                    G.nodes[node]["predicates_freq"][pred_col] = current + freq

        # Process join edges
        for edge in query.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            condition = edge.get("condition")

            if not source or not target or source == target:
                continue

            # Normalization for avoiding duplicates edges
            if source > target:
                source, target = target, source

            if condition and "=" in condition:
                parts = [p.strip() for p in condition.split("=")]
                normalized_condition = " = ".join(sorted(parts))
            else:
                normalized_condition = condition

            if not G.has_edge(source, target):
                G.add_edge(source, target, weight=0, conditions={})

            edge_data = G[source][target]
            edge_data["conditions"][normalized_condition] = edge_data["conditions"].get(normalized_condition, 0) + 1
            edge_data["weight"] = sum(edge_data["conditions"].values())
            edge_data["condition"] = max(edge_data["conditions"], key=edge_data["conditions"].get)

    # Optional entity resolution using Spark
    if spark is not None:
        for node in G.nodes():
            sample_values = {}
            columns_to_resolve = list(G.nodes[node].get("columns_freq", {}).keys())

            for col in columns_to_resolve:
                try:
                    query = f"SELECT DISTINCT `{col}` FROM `{node}` WHERE `{col}` IS NOT NULL LIMIT 3"
                    rows = spark.sql(query).collect()
                    vals = [str(row[0]) for row in rows if row[0] is not None]
                    if vals:
                        sample_values[col] = vals
                except Exception:
                    continue

            G.nodes[node]["sample_values"] = sample_values
    else:
        print("\nWarning: No Spark session provided. Entity Resolution skipped.")

    return G

def save_graph(G, db_name):
    filename = get_graph_path(db_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nGraph successfully saved to {filename}")

def verify_graph(G):
    print(f"\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}\n")

    for node, data in G.nodes(data=True):
        print(f"[{node}]")
        print(f"  Cols: {data.get('columns_freq')}")
        print(f"  Ops: {data.get('operations_freq')}")
        print(f"  Aggs: {data.get('aggregations_freq')}")
        print(f"  Preds: {data.get('predicates_freq')}")
        if data.get('sample_values'):
            print(f"  Samples: {data.get('sample_values')}")
        print()

    for u, v, data in G.edges(data=True):
        print(f"[{u}] -> [{v}] (w:{data.get('weight')})")
        print(f"  Main condition: {data.get('condition')}")
        print(f"  All conditions (freq): {data.get('conditions')}\n")

        
if __name__ == "__main__":
    # Ensure project root is in the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--db", type=str, required=True)
    args = arg_parser.parse_args()
    DB_NAME = args.db

    try:
        # Optional Spark dependencies
        from utils import ensure_sqlite_jdbc_driver
        from load_db import load_tables
        from spark_nl import get_spark_session

        # Load JDBC driver for SQLite
        jdbc_jar_path = ensure_sqlite_jdbc_driver()

        # Initialize Spark session
        spark = get_spark_session(extra_configs={
            "spark.jars": jdbc_jar_path,
            "spark.driver.extraClassPath": jdbc_jar_path,
        })

        # Load database tables into Spark
        load_tables(spark, DB_NAME)

        # Build property graph with entity resolution enabled
        G = build_graph(DB_NAME, spark=spark)

        # Stop Spark session
        spark.stop()

    except ImportError:
        # Fallback when Spark is not available
        print("Could not load Spark dependencies. Building graph without Entity Resolution.")
        G = build_graph(DB_NAME)

    # Save graph to disk
    save_graph(G, DB_NAME)

    # Print and visualize graph
    verify_graph(G)