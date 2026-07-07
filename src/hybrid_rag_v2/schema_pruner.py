import json
from difflib import SequenceMatcher
from typing import Dict, List, Any

def _lexical_overlap(nl_query: str, columns: List[str]) -> float:
    """Calculates simple lexical overlap between the question and the table columns."""
    q_tokens = set(nl_query.lower().split())
    if not columns:
        return 0.0
    
    hits = 0
    for c in columns:
        # We check if any word in the query is highly similar to the column name
        if any(SequenceMatcher(None, c.lower(), tok).ratio() > 0.75 for tok in q_tokens):
            hits += 1
            
    return hits / len(columns)

def _llm_verify_discard(nl_query: str, candidates: List[str], graph: Any, llm: Any) -> List[str]:
    """Second pass (Tier 2): LLM explicitly verifies if a table is required based on its columns."""
    survivors = []
    for t in candidates:
        if not graph.has_node(t):
            continue
            
        cols = list(graph.nodes[t].get("columns_freq", {}).keys())[:8]
        prompt = (
            f"Question: \"{nl_query}\"\n"
            f"Table \"{t}\" has columns: {cols}\n"
            f"Is at least one of these exact columns required to compute the SELECT, WHERE, "
            f"or JOIN key of the answer? Answer ONLY 'YES' or 'NO'."
        )
        
        try:
            # Assumes the LLM object has an 'invoke' method (like LangChain)
            resp = llm.invoke(prompt).content.strip().upper()
            if resp.startswith("YES"):
                survivors.append(t)
        except Exception as e:
            print(f"Warning: LLM verify failed for table {t}: {e}")
            survivors.append(t) # If LLM fails, we keep the table to be safe
            
    # Return tables that were truly discarded (not in survivors)
    return [t for t in candidates if t not in survivors]

def prune_candidate_tables(nl_query: str, candidate_tables: List[str], graph: Any, 
                           all_pair_paths: Dict, llm: Any = None, 
                           overlap_threshold: float = 0.05) -> Dict[str, List[str]]:
    """
    Main pruning function.
    Returns a dict: {"required": [...], "bridge": [...], "discarded": [...]}
    """
    required, maybe_discard = [], []
    
    # Tier 1: Lexical check
    for t in candidate_tables:
        cols = list(graph.nodes[t].get("columns_freq", {}).keys()) if graph.has_node(t) else []
        score = _lexical_overlap(nl_query, cols)
        
        if score >= overlap_threshold:
            required.append(t)
        else:
            maybe_discard.append(t)
            
    # Identify bridge tables (tables needed to connect required tables)
    bridge = set()
    for (u, v), path in all_pair_paths.items():
        if u in required and v in required:
            for node in path:
                if node in maybe_discard:
                    bridge.add(node)
                    
    discarded = [t for t in maybe_discard if t not in bridge]
    
    # Tier 2: LLM Verification for the discarded tables (Optional but recommended)
    if llm and discarded:
        discarded = _llm_verify_discard(nl_query, discarded, graph, llm)
        
    # Re-calculate bridge in case Tier 2 promoted a discarded table back to required
    # For now, we will simply append the survivors back to required
    survivors = [t for t in maybe_discard if t not in bridge and t not in discarded]
    required.extend(survivors)
        
    return {
        "required": list(set(required)),
        "bridge": sorted(list(bridge)),
        "discarded": discarded
    }