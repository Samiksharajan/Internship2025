import os
import json

# Define the correct folder path
folder_path = "/Users/samiksharajan/Documents/Project/embeddings_output"

# Output JSON file path
output_file = os.path.join(folder_path, "merged_embeddings.json")

# Dictionary to store merged data
merged_data = {}

# Iterate through all embedding files
for file_name in sorted(os.listdir(folder_path)):  # Sort for consistency
    if file_name.endswith("_embeddings.txt"):
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as file:
                data = file.readlines()
            
            # Store in dictionary (using file name as key without extension)
            merged_data[file_name.replace("_embeddings.txt", "")] = [line.strip() for line in data]
        
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

# Save as JSON
try:
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(merged_data, json_file, indent=4)

    print(f"Merged embeddings saved to {output_file}")
except Exception as e:
    print(f"Error saving JSON file: {e}")
