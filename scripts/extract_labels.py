
import csv
import json
import os

def extract_labels():
    input_file = r'c:\Users\Hyde\Desktop\DDI\DDI\data\drugbank.tab'
    output_file = r'c:\Users\Hyde\Desktop\DDI\DDI\data\processed\label_names.json'
    
    print(f"Reading from {input_file}...")
    mapping = {}
    
    try:
        # Check if file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file does not exist: {input_file}")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            # First handle potential quoting issues manually or rely on csv module
            reader = csv.DictReader(f, delimiter='\t')
            row_count = 0
            for row in reader:
                try:
                    # The Y column acts as the key, Map as the value
                    if 'Y' in row and 'Map' in row:
                        y_val = row['Y'].strip()
                        if y_val.isdigit():
                            y = int(y_val)
                            desc = row['Map']
                            mapping[y] = desc
                            row_count += 1
                except ValueError:
                    continue
        
        print(f"Found {len(mapping)} unique interaction types from {row_count} rows.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4)
            
        print(f"Successfully saved label map to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_labels()
