import pandas as pd
import os
import json

# Path to the Excel file
excel_path = "/Users/samiksharajan/Downloads/Training Data for Cleartrust Labels.xlsx"
output_folder = "/Users/samiksharajan/Downloads/json_templates"

# Load the Excel file
xls = pd.ExcelFile(excel_path)

# Create a mapping: cleaned sheet names → original sheet names
sheet_mapping = {sheet.strip().replace(" ", ""): sheet for sheet in xls.sheet_names}

# Define sheets to exclude
exclude_sheets = {"PossibleQueries", "Traps", "GeneralQueries"}

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

print("📄 Raw Sheet Names:", xls.sheet_names)
print("✅ Cleaned Sheet Names:", list(sheet_mapping.keys()))

# Process each sheet except the excluded ones
for clean_name, original_name in sheet_mapping.items():  # Ensure this loop is present
    if clean_name in exclude_sheets:
        print(f"🚫 Skipping {original_name}: Excluded from processing.")
        continue

    try:
        # Read the sheet using the original name
        df = pd.read_excel(xls, sheet_name=original_name, header=0)

        # Debugging: Print Data Info
        print(f"🔍 Processing {original_name} - Total Rows: {len(df)}")
        print(f"📊 Columns in {original_name}: {df.columns.tolist()}")  # Print column names

        # Convert to structured JSON format
        json_data = []
        if len(df.columns) >= 3:  # Ensure at least three columns exist
            for _, row in df.iloc[1:].iterrows():  # Skip first row
                template_type = str(row[df.columns[0]]).strip() if pd.notna(row[df.columns[0]]) else "UNKNOWN"  # First column
                mapped_id = str(row[df.columns[1]]).strip() if pd.notna(row[df.columns[1]]) else "UNKNOWN"  # Second column
                synonyms = [str(v).strip() for v in str(row[df.columns[2]]).split(",") if str(v).strip()]  # Third column

                json_data.append({
                    "template_type": template_type,
                    "mapped_id": mapped_id,
                    "synonyms": synonyms
                })

        # Save JSON file
        json_path = os.path.join(output_folder, f"{clean_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)

        print(f"✅ Saved {json_path}")

    except Exception as e:  # Catch all exceptions
        print(f"❌ Error processing {original_name}: {e}")

print("🎉 All selected JSON files have been generated successfully!")

