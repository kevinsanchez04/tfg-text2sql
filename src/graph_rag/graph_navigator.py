import networkx as nx
import matplotlib.pyplot as plt
import json

def graph_navigator(t1, t2):
    with open('test_ast.json') as json_file:
        ast = json.load(json_file)
    
    G = nx.Graph()


if __name__ == "__main__":
    prompt = graph_navigator()
    #verify_graph(G)
    print(f"Prompt generated:\n\n{prompt}")