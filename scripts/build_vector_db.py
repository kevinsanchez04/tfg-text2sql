import chromadb
import json

# Inicialize client
client = chromadb.PersistentClient(path="./db/vector_store")

# Create or loading the collection
collection = client.get_or_create_collection(name="toxicology_queries")

DEV_FILE = "db/bird-1/dev.json"

# Read dev.json
with open(DEV_FILE, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

docs_nl = []    # ["What is the most common bond type?", ...]
metadata = []   # [{"query_id": 195, "golden_sql": "SELECT ..."}, ...]
unique_ids = [] # ["195", ...]

for item in dev_data:
    if item.get("db_id") == "toxicology":
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