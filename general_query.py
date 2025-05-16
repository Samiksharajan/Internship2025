import pandas as pd
import os

# Define file paths
file_path = "/Users/samiksharajan/Downloads/Training Data for Cleartrust Labels.xlsx"  # Update with actual path
sheet_name = "General Queries"  # Sheet name
output_file = "/Users/samiksharajan/Documents/Project/general_query_output.json"  # Full path to save JSON

try:
    # Read the specific sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Check if DataFrame is empty
    if df.empty:
        print("⚠️ Error: The DataFrame is empty. Check the Excel sheet content.")
    else:
        # Convert to JSON
        json_output = df.to_json(orient="records", indent=4)

        # Print a preview of the JSON
        print("✅ JSON Preview:\n", json_output[:500])

        # Save JSON to a file
        with open(output_file, "w") as json_file:
            json_file.write(json_output)

        print(f"🎉 Success! JSON file saved at: {output_file}")

except Exception as e:
    print(f"❌ Error: {e}")

