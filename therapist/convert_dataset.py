import json
import argparse

def convert_dataset(input_file_path, output_file_path):
    """Reads a JSON file, converts specified fields, and writes to a new JSON file."""
    try:
        data = []
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    item = json.loads(line.strip()) # Use json.loads for each line
                    data.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line due to JSON decode error: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}")
        return

    converted_data = []
    if isinstance(data, list):
        for item in data:
            new_item = {}
            new_item['input'] = ""
            if 'Context' in item:
                new_item['instruction'] = item.pop('Context')
            if 'Response' in item:
                new_item['output'] = item.pop('Response')
            # Add any other fields from the original item
            new_item.update(item)
            converted_data.append(new_item)
            
    # The data is now guaranteed to be a list of dictionaries if successfully parsed
    # So, we remove the check for isinstance(data, dict) and the subsequent else block
    # as the new parsing logic handles line-by-line objects into a list.

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(converted_data, outfile, indent=2)
        print(f"Successfully converted dataset and saved to {output_file_path}")
    except IOError:
        print(f"Error: Could not write to output file at {output_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON dataset fields.')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file.')
    parser.add_argument('output_file', type=str, help='Path to the output JSON file.')

    args = parser.parse_args()

    convert_dataset(args.input_file, args.output_file)
