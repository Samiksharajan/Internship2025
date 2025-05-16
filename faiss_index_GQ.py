import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

###############################################################################
# 1) Load your JSON data (assumes each item has "Query", "Response", "Query_type")
###############################################################################
input_file = "/Users/samiksharajan/Downloads/merged_GQ_output_final.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

###############################################################################
# 2) Build a Faiss index with a bi-encoder
###############################################################################
model_bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert each "Query" to an embedding
queries = [item["Query"] for item in data]
embeddings = model_bi.encode(queries, show_progress_bar=True)
embeddings = np.array(embeddings, dtype="float32")  # Faiss needs float32

# Create a Flat L2 index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Store metadata so we can refer back to the original items
metadata = [
    {
        "Index": i,
        "Query": item["Query"],
        "Response": item.get("Response", ""),
        "Query_type": item.get("Query_type", "")
    }
    for i, item in enumerate(data)
]

def search_faiss(query_text, k=5):
    """
    Retrieve top k results from Faiss using the bi-encoder embeddings.
    Returns a list of dicts with distance + original metadata.
    """
    query_vec = model_bi.encode([query_text], show_progress_bar=False)
    query_vec = np.array(query_vec, dtype="float32")
    
    # Faiss search
    distances, indices = index.search(query_vec, k)
    
    # Collect results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        item = metadata[idx].copy()
        item["distance"] = float(dist)  # L2 distance from the query
        results.append(item)
    return results

###############################################################################
# 3) Cross-Encoder for Reranking
###############################################################################
model_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank(query_text, top_k=5):
    """
    1) Get top_k results from Faiss via bi-encoder embeddings.
    2) Rerank them with a cross-encoder for higher accuracy.
    3) Return the list of candidates sorted by cross-encoder score (descending).
    """
    # 1) Bi-encoder retrieval
    candidates = search_faiss(query_text, k=top_k)
    
    # 2) Prepare pairs for cross-encoder
    cross_inp = [(query_text, c["Query"]) for c in candidates]
    
    # 3) Get cross-encoder scores
    scores = model_cross.predict(cross_inp)  # Higher = more relevant
    for c, s in zip(candidates, scores):
        c["cross_score"] = float(s)
    
    # Sort by cross-encoder score in descending order
    candidates.sort(key=lambda x: x["cross_score"], reverse=True)
    return candidates

###############################################################################
# 4) Interactive loop: show all top_k matches
###############################################################################
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter your query (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Get top k results (you can adjust k as needed)
        top_k = 5
        results = rerank(user_input, top_k=top_k)
        
        print(f"\nTop {top_k} Reranked Results:")
        for i, r in enumerate(results, start=1):
            print(f"{i}. Query: {r['Query']}")
            print(f"   Response: {r['Response']}")
            print(f"   Query_type: {r['Query_type']}")
            print(f"   Distance (bi-encoder): {r['distance']:.4f}")
            print(f"   Cross-Encoder Score: {r['cross_score']:.4f}")
            print()
