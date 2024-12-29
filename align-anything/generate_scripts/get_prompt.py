import json

# Input and output file paths
input_file = 'train.jsonl'  # Replace with the path to your .jsonl file
output_file = 'output.json'  # Replace with the desired output .json file path

# List to hold the processed prompts
prompts = []

# Open the .jsonl file and read line by line
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        # Parse each line (it’s a JSON object)
        data = json.loads(line)
        
        # Extract the "prompt" field and add it to the list
        if 'prompt' in data:
            prompts.append({"prompt": data['prompt']})

# Write the extracted prompts to a new .json file
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(prompts, outfile, indent=4)

print(f"Extracted prompts have been saved to {output_file}")
