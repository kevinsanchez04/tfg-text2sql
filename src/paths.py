# src/paths.py
import os

def get_vector_store_path(db_name: str) -> str:
    return os.path.join("db", "vector_store", db_name)

def get_vector_collection_name(db_name: str) -> str:
    return f"{db_name}_queries"

def get_graph_path(db_name: str) -> str:
    return os.path.join("data", "graphs", db_name, "property_graph.json")

def get_ast_path(db_name: str) -> str:
    return os.path.join("data", "ast", db_name, "ast.json")

def get_logs_folder(db_name: str) -> str:
    return os.path.join("logs", db_name)

def get_plots_folder(db_name: str) -> str:
    return os.path.join("data", "plots", db_name)