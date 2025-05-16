import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load JSON data
with open("/Users/samiksharajan/Documents/Project/merged_output.json", "r") as f:
    data = json.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create mappings
synonym_to_id = {}
synonyms_list = []

if isinstance(data, dict):
    for key, value in data.items():
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and "mapped_id" in entry:
                    mapped_id = entry["mapped_id"]
                    for synonym in entry.get("synonyms", []):
                        synonym_lower = synonym.lower()
                        synonym_to_id[synonym_lower] = mapped_id
                        synonyms_list.append(synonym_lower)
elif isinstance(data, list):
    for entry in data:
        if isinstance(entry, dict) and "mapped_id" in entry:
            mapped_id = entry["mapped_id"]
            for synonym in entry.get("synonyms", []):
                synonym_lower = synonym.lower()
                synonym_to_id[synonym_lower] = mapped_id
                synonyms_list.append(synonym_lower)
else:
    raise ValueError("Unexpected JSON format in merged_output.json")

# Encode all synonyms
synonym_embeddings = model.encode(synonyms_list)
synonym_embeddings = np.array(synonym_embeddings, dtype=np.float32)

# Create FAISS index
dimension = synonym_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(synonym_embeddings)

# Save the index
faiss.write_index(faiss_index, "/Users/samiksharajan/Documents/Project/synonym_faiss.index")

# Save mappings (to ensure correct order)
with open("/Users/samiksharajan/Documents/Project/merged_embeddings.json", "w") as f:
    json.dump({"synonyms": synonyms_list, "synonym_to_id": synonym_to_id}, f)

print("✅ FAISS index created and saved successfully!")
