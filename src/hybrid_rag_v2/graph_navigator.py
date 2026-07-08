import networkx as nx
import json
import os
from typing import List, Tuple, Dict, Any

from hybrid_rag_v2.schema_pruner import prune_candidate_tables

def load_graph(graph_path: str = "db/bird-1/toxicology/property_graph.json") -> nx.Graph:
    """Loads the property graph from JSON."""
    G = nx.Graph()
    if not os.path.exists(graph_path):
        print(f"Warning: Graph file not found at {graph_path}")
        return G
        
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for node, attrs in data.get("nodes", {}).items():
        G.add_node(node, **attrs)
    for edge in data.get("edges", []):
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))
        
    return G

def build_schema_map(pruned: Dict[str, List[str]], graph: nx.Graph) -> str:
    """
    Substitutes the raw metadata block with a compact map without logic_freq numbers.
    Reads pre-computed sample values directly from the graph properties for Entity Resolution.
    """
    lines = ["### SCHEMA MAP (curated) ###"]
    
    for tag, tables in [("REQUIRED", pruned["required"]), ("BRIDGE-ONLY", pruned["bridge"])]:
        for t in tables:
            if not graph.has_node(t):
                continue
                
            cols = list(graph.nodes[t].get("columns_freq", {}).keys())[:6]
            table_def = f"[{tag}] {t}({', '.join(cols)})"
            lines.append(table_def)
            
            if tag == "REQUIRED":
                samples = graph.nodes[t].get("sample_values", {})
                if samples:
                    lines.append("    CRITICAL - EXACT VALUES TO USE IN 'WHERE' CLAUSE:")
                    for c, vals in samples.items():
                        lines.append(f"      - {c}: You MUST map natural language to one of these: {vals}")
                        
    if pruned["bridge"]:
        lines.append(
            "\nNote: BRIDGE-ONLY tables exist solely to connect REQUIRED tables via JOIN. "
            "Do not select or filter on their columns unless the question explicitly asks for them."
        )
        
    return "\n".join(lines)

def graph_navigator(tables: List[str], nl_query: str, llm: Any = None) -> Tuple[str, Dict]:
    """
    Generates the structural context based on the query and candidate tables.
    Returns the formatted prompt string and a dictionary of the best paths.
    """
    G = load_graph()
    
    # Ensure tables exist in the graph
    valid_tables = [t for t in tables if G.has_node(t)]
    
    # Pre-calculate all shortest paths between valid tables to find bridges
    all_pair_paths = {}
    for i in range(len(valid_tables)):
        for j in range(i + 1, len(valid_tables)):
            u, v = valid_tables[i], valid_tables[j]
            try:
                # Assuming weight represents frequency, we want the "lightest" cost.
                path = nx.shortest_path(G, source=u, target=v) 
                all_pair_paths[(u, v)] = path
            except nx.NetworkXNoPath:
                continue

    # Context Pruning
    pruned_dict = prune_candidate_tables(
        nl_query=nl_query,
        candidate_tables=valid_tables,
        graph=G,
        all_pair_paths=all_pair_paths,
        llm=llm
    )
    
    # Filter the best paths to only include those between required/bridge tables
    final_nodes = set(pruned_dict["required"] + pruned_dict["bridge"])
    best_path_per_pair = {}
    for (u, v), path in all_pair_paths.items():
        if u in final_nodes and v in final_nodes:
             best_path_per_pair[(u, v)] = path

    # Build the text prompt (Schema Map) reading the pre-computed values
    prompt_text = build_schema_map(pruned_dict, G)
    
    return prompt_text, best_path_per_pair