import networkx as nx
from networkx.readwrite import json_graph
import json
import itertools
import os

JOIN_ALTERNATIVE_THRESHOLD = 0.3  # The second condition must have at least 30% of the first condition's weight

def load_graph(filename='data/graphs/property_graph.json'):
    
    if not os.path.exists(filename):
        print(f"Warning: Graph file not found at {filename}")
        return nx.Graph()
    
    with open(filename, 'r') as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


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


def graph_navigator(tables: list, top_k_meta=3):
    G = load_graph()
    
    # filter out hallucinated or missing tables
    valid_tables = [t for t in tables if G.has_node(t)]
    
    if not valid_tables:
        return "No valid historical tables identified."

    prompt = ""
    best_paths_nodes = set()
    join_conditions = set()
    
    if len(valid_tables) == 1:
        # no joins needed, just extract metadata for the single table
        prompt += "### SINGLE TABLE QUERY ###\nNo JOINs required.\n"
        best_paths_nodes.add(valid_tables[0])
    
    else:
        prompt += "### SUGGESTED JOIN CONDITIONS ###\n"
        
        # generate all unique pairs of requested tables
        pairs = list(itertools.combinations(valid_tables, 2))
        
        for u, v in pairs:
            paths = list(nx.all_simple_paths(G, source=u, target=v, cutoff=3))
            if not paths:
                continue
            
            # evaluate paths for current pair
            max_w = -1
            best_path = []
            for p in paths:
                weight_p = sum(G[p[i]][p[i+1]].get("weight", 0) for i in range(len(p)-1))
                if weight_p > max_w:
                    max_w = weight_p
                    best_path = p
            
            # extract conditions and nodes from the winning path
            if best_path:
                best_paths_nodes.update(best_path)
                for i in range(len(best_path) - 1):
                    n1, n2 = best_path[i], best_path[i+1]
                    edge_data = G[n1][n2]
                    # using set to ensure overlapping paths don't create duplicate rules
                    t_a, t_b = sorted([n1, n2])

                    alt_conditions = _format_join_conditions(edge_data)
                    if len(alt_conditions == 1):
                        join_conditions.add(f"- For joining {t_a} and {t_b} use: {alt_conditions[0]}")
                    else:
                        alt_str = " OR ".join(alt_conditions)
                        join_conditions.add(
                            f"- For joining {t_a} and {t_b}, multiple valid conditions exist "
                            f"depending on context, choose the correct one: {alt_str}"
                        )

        if join_conditions:
            prompt += "\n".join(sorted(join_conditions)) + "\n"
        else:
            prompt += "No historical JOIN paths found.\n"
            best_paths_nodes.update(valid_tables)

    # append metadata for all nodes involved in the resolution
    prompt += "\n### HISTORICAL METADATA ###\n"
    
    for node in sorted(best_paths_nodes):
        cols = G.nodes[node].get("columns_freq", {})
        logic = G.nodes[node].get("logic_freq", {})
        
        top_cols = sorted(cols.items(), key=lambda x: x[1], reverse=True)[:top_k_meta]
        top_logic = sorted(logic.items(), key=lambda x: x[1], reverse=True)[:top_k_meta]
        
        prompt += f"\nTable: [{node}]\n"
        
        if top_cols:
            cols_str = ", ".join([f"{c[0]} (freq: {c[1]})" for c in top_cols])
            prompt += f"  - Top Columns: {cols_str}\n"
        else:
            prompt += "  - Top Columns: None\n"
            
        if top_logic:
            prompt += "  - Common Filters:\n"
            for l in top_logic:
                prompt += f"    * {l[0]} (freq: {l[1]})\n"
        else:
            prompt += "  - Common Filters: None\n"
    
    return prompt

if __name__ == "__main__":
    # test scenarios
    print("--- TEST 1: Two tables ---")
    print(graph_navigator(['molecule', 'connected']))
    
    print("\n--- TEST 2: Three tables ---")
    print(graph_navigator(['molecule', 'atom', 'connected']))
    
    print("\n--- TEST 3: One table ---")
    print(graph_navigator(['molecule']))