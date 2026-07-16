import chromadb
import json
import argparse
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from paths import get_vector_store_path, get_vector_collection_name

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, required=True, help="db_id in dev.json (e.g. toxicology)")
args = parser.parse_args()
DB_NAME = args.db

# Inicialize client
client = chromadb.PersistentClient(path=get_vector_store_path(DB_NAME))

# Create or loading the collection
collection = client.get_or_create_collection(name=get_vector_collection_name(DB_NAME))

DEV_FILE = "db/bird-1/dev.json"

# Read dev.json
with open(DEV_FILE, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

docs_nl = []    # ["What is the most common bond type?", ...]
metadata = []   # [{"query_id": 195, "golden_sql": "SELECT ..."}, ...]
unique_ids = [] # ["195", ...]

for item in dev_data:
    if item.get("db_id") == DB_NAME:
        q_id = item["question_id"]
        docs_nl.append(item["question"])
        metadata.append({
            "query_id": q_id,
            "golden_sql": item["SQL"]
        })
        # String ID for chromadb
        unique_ids.append(str(q_id))

print(f"Inserting {len(docs_nl)} queries in ChromaDB")

collection.add(documents=docs_nl, metadatas=metadata, ids=unique_ids)

print("Vector database created successfully")