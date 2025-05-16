import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load JSON data
json_file = "merged_output.json"
try:
    with open(json_file, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{json_file}' was not found. Ensure it's in the correct directory.")
    exit(1)

# Extract synonyms and mapped IDs
synonym_to_id = {}
all_synonyms = []
mapped_ids = []

for key, value_list in data.items():
    if not isinstance(value_list, list):
        continue

    for value in value_list:
        if not isinstance(value, dict) or "mapped_id" not in value or "synonyms" not in value:
            continue

        mapped_id = value["mapped_id"]
        synonyms = value["synonyms"]

        for synonym in synonyms:
            synonym_to_id[synonym] = mapped_id
            all_synonyms.append(synonym)
            mapped_ids.append(mapped_id)

# Convert synonyms to embeddings
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
synonym_embeddings = model.encode(all_synonyms, convert_to_numpy=True)

# Create FAISS index
dimension = synonym_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(synonym_embeddings)

def find_mapped_id(query):
    """Find the closest mapped_id for a given query using FAISS"""
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, 1)
    closest_synonym = all_synonyms[indices[0][0]]
    return synonym_to_id.get(closest_synonym, "Mapped ID not found")

# Example usage
if __name__ == "__main__":
    while True:
        query = input("\nEnter a synonym (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        mapped_id = find_mapped_id(query)
        print(f"Mapped ID for '{query}': {mapped_id}")
