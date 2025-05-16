import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Excel file
file_path = "/Users/samiksharajan/Downloads/Training Data for Cleartrust Labels.xlsx"
sheets_to_exclude = ["General Queries", "Possible Queries", "Traps"]

# Load all sheets
xls = pd.ExcelFile(file_path)
all_sheets = xls.sheet_names
included_sheets = [name for name in all_sheets if name not in sheets_to_exclude]

print("Available sheets:", all_sheets)
print("Processing sheets:", included_sheets)

sheets = {name: xls.parse(name) for name in included_sheets}

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS
embedding_size = 384  # Depends on the model used
index = faiss.IndexFlatL2(embedding_size)

for sheet_name, df in sheets.items():
    print(f"Processing sheet: {sheet_name}")

    if df.shape[1] < 3:
        print(f"Skipping {sheet_name}, not enough columns.")
        continue

    synonyms = df.iloc[:, 2].dropna().astype(str).tolist()

    if not synonyms:
        print(f"Skipping {sheet_name}, no valid data in column 3.")
        continue

    # Generate embeddings
    embeddings = model.encode(synonyms, convert_to_numpy=True)

    # Add to FAISS index
    index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "synonym_embeddings.index")
print("✅ FAISS Embeddings generated and saved successfully!")
