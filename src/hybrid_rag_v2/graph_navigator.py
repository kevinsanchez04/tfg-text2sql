import math
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import numpy as np
from paths import get_graph_path


JOIN_ALTERNATIVE_THRESHOLD = 0.3 # The second condition must have at least 30% of the first condition's weight


def calculate_similarity(vector1, vector2):
    """Calculates cosine similarity between two embeddings to measure semantic closeness."""
    if not vector1 or not vector2:
        return 0

    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def load_graph(db_name):
    file_path = get_graph_path(db_name)
    if not os.path.exists(file_path):
        print(f"Warning: Graph file not found: {file_path}")
        return nx.Graph()

    with open(file_path, "r") as file:
        graph_data = json.load(file)

    # Reconstruct the NetworkX graph object from the saved dictionary format
    return nx.Graph(json_graph.node_link_graph(graph_data))


def _format_join_conditions(edge_data):
    """Return the join condition(s) for an edge, including common alternatives."""

    conditions = edge_data.get("conditions")

    if not conditions:
        return [edge_data.get("condition", "Unknown")]

    ranked = sorted(conditions.items(), key=lambda kv: kv[1], reverse=True)
    top_cond, top_freq = ranked[0]

    selected = [top_cond]
    for cond, freq in ranked[1:]:
        # Include alternative conditions if they are frequent enough
        if freq / top_freq >= JOIN_ALTERNATIVE_THRESHOLD:
            selected.append(cond)

    return selected

def graph_navigator(nl_query: str,
                    tables: list,
                    db_name: str,
                    embedder=None,
                    top_k_metadata=3):
    """
    Builds the structural context for the LLM. 
    It finds the best JOIN paths connecting the requested tables and appends 
    historical metadata to guide the SQL generation.
    """
    graph = load_graph(db_name)

    max_edge_weight = max([data.get("weight", 0) for _, _, data in graph.edges(data=True)]) if graph.number_of_edges() > 0 else 1

    available_tables = [
        table for table in tables
        if graph.has_node(table)
    ]

    if not available_tables:
        return "No valid historical tables identified."

    prompt = ""
    relevant_nodes = set()
    join_rules = set()

    query_embedding = (
        embedder.embed_query(nl_query)
        if embedder else None
    )

    if len(available_tables) == 1:
        # Fast exit for simple queries to prevent unnecessary graph traversal
        prompt += "### SINGLE TABLE QUERY ###\n"
        prompt += "No JOINs required.\n"
        relevant_nodes.add(available_tables[0])

    else:
        prompt += "### SUGGESTED JOIN CONDITIONS ###\n"

        # Incremental subgraph construction to ensure all tables are connected 
        # without leaving isolated components. We start with the first table and grow.
        connected_tables = {available_tables[0]}
        pending_tables = set(available_tables[1:])

        while pending_tables:
            best_path = []
            best_score = -float('inf')
            selected_target = None

            for source_table in connected_tables:
                for target_table in pending_tables:
                    # Look for short paths (cutoff=3) to prevent overly complex routing
                    paths = list(nx.all_simple_paths(graph,source=source_table,target=target_table,cutoff=3))
                    
                    for path in paths:
                        # Historical socre (log-minmax normalized 0.0 to 1.0)
                        historical_sum  = 0
                        num_edges = len(path) - 1

                        for i in range(num_edges):
                            raw_weight = graph[path[i]][path[i+1]].get("weight", 0)
                            normalized_weight = math.log1p(raw_weight) / math.log1p(max_edge_weight)
                            historical_sum += normalized_weight

                        historical_score = historical_sum / num_edges
                        
                        # Semantic score (-1.0 to 1.0)
                        # Positive values boost the score, while negative values penalize it
                        semantic_score = 0
                        if embedder and query_embedding:
                            # Evaluate how well the tables in this path match the user's query
                            path_text = " ".join(path)
                            path_embedding = embedder.embed_query(path_text)
                            semantic_score = calculate_similarity(query_embedding,path_embedding)

                        # Hybrid routing score: combines historical frequency with semantic relevance
                        # to reduce the bias toward highly connected tables
                        ALPHA = 0.4
                        total_score = (ALPHA*historical_score) + ((1-ALPHA) * semantic_score)
                        
                       # Penalize longer paths to favor shorter ones
                        total_score = total_score * (0.95 ** (num_edges - 1))

                        if total_score > best_score:
                            best_score = total_score
                            best_path = path
                            selected_target = target_table

            if best_path:
                # Absorb the found path into our connected component
                relevant_nodes.update(best_path)
                connected_tables.update(best_path)
                pending_tables.remove(selected_target)

                # Extract the specific SQL conditions used historically between these nodes
                for i in range(len(best_path) - 1):
                    left_table = best_path[i]
                    right_table = best_path[i + 1]
                    edge_data = graph[left_table][right_table]
                    # Sort table names to avoid duplicate rules like A-B and B-A
                    table_a, table_b = sorted([left_table, right_table])

                    alt_conditions = _format_join_conditions(edge_data)
                    if len(alt_conditions) == 1:
                        join_rules.add(
                            f"- For joining {table_a} and {table_b} use: {alt_conditions[0]}"
                        )
                    else:
                        alt_str = " OR ".join(alt_conditions)
                        join_rules.add(
                            f"- For joining {table_a} and {table_b}, multiple valid conditions exist "
                            f"depending on context, choose the correct one: {alt_str}"
                        )
            else:
                # Fallback if the graph is disconnected (no path exists)
                relevant_nodes.update(pending_tables)
                break

        if join_rules:
            prompt += "\n".join(sorted(join_rules)) + "\n"
        else:
            prompt += "No historical JOIN paths found.\n"

    prompt += "\n### HISTORICAL METADATA ###\n"

    # Inject contextual hints into the prompt so the LLM uses the correct schema logic
    for table in sorted(relevant_nodes):
        columns = graph.nodes[table].get("columns_freq", {})
        operations = graph.nodes[table].get("operations_freq", {})
        aggregations = graph.nodes[table].get("aggregations_freq", {})
        predicates = graph.nodes[table].get("predicates_freq", {})
        samples = graph.nodes[table].get("sample_values", {})

        top_columns = sorted(columns.items(), key=lambda item: item[1], reverse=True)[:top_k_metadata]
        
        # Filter out 0-frequency operations and sort by usage
        active_operations = [k for k, v in sorted(operations.items(), key=lambda item: item[1], reverse=True) if v > 0]
        active_aggregations = [k for k, v in sorted(aggregations.items(), key=lambda item: item[1], reverse=True) if v > 0]
        active_predicates = [k for k, v in sorted(predicates.items(), key=lambda item: item[1], reverse=True) if v > 0]

        prompt += f"\nTable: [{table}]\n"

        if top_columns:
            prompt += "  - Frequently referenced columns: " + ", ".join(column for column, _ in top_columns) + "\n"

        if samples:
            # Crucial for preventing hallucinations on literal values (e.g. knowing bond type is '#' not 'Triple')
            prompt += "  - Sample data formats:\n"
            for col, vals in samples.items():
                if vals:
                    prompt += f"    * {col}: {vals}\n"

        prompt += "  - Historical SQL patterns:\n"
        
        has_patterns = False
        if active_predicates:
            prompt += f"    * WHERE clauses frequently use: {', '.join(active_predicates[:top_k_metadata])}\n"
            has_patterns = True
        if active_operations:
            prompt += f"    * Used operators: {', '.join(active_operations)}\n"
            has_patterns = True
        if active_aggregations:
            prompt += f"    * Aggregations: {', '.join(active_aggregations)}\n"
            has_patterns = True
            
        if not has_patterns:
            prompt += "    * No structural patterns identified.\n"

    return prompt