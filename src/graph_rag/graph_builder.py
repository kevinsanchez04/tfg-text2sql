import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import json

def build_graph(filename='data/ast/test_ast.json'):
    with open(filename) as json_file:
        ast = json.load(json_file)

    G = nx.Graph()

    for query in ast:
        query_nodes = query.get("nodes", [])
        
        # Process nodes and update frequency metadata for the Property Graph
        for node in query_nodes:
            if not G.has_node(node):
                G.add_node(node, columns_freq={}, logic_freq={})
            
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

    return G

def save_graph(G, filename='data/graphs/propert_graph.json'):
    data = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Graph successfully saved to {filename}")

def verify_graph(G):
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}\n")
    
    for node, data in G.nodes(data=True):
        print(f"[{node}]")
        print(f"  Cols: {data.get('columns_freq')}")
        print(f"  Logic: {data.get('logic_freq')}\n")

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
    G = build_graph()
    save_graph(G)
    #verify_graph(G)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")