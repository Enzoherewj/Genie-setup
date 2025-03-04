import json
import pandas as pd
import re
from vllm import LLM, SamplingParams

# Initialize the GENIE model
model = LLM(model='THUMedInfo/GENIE_en_8b', tensor_parallel_size=1)
PROMPT_TEMPLATE = "Human:\nBelow is a clinical note. Please analyze it and provide a summary of the patient's condition, key findings, and recommended next steps.\n\n{query}\n\nAssistant:"

# Set sampling parameters with a very large max token value
temperature = 0.7
max_new_token = 2048  # Using a much larger value
sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_token)

# Function to read clinical notes from CSV file
def read_notes_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    notes = df['text'].tolist()
    # Limit to first 10 notes
    return notes[:10]

# Function to format JSON with proper indentation (from simple_format.py)
def format_json_response(text):
    """Format JSON content with proper indentation if possible."""
    try:
        # Try to fix truncated JSON by adding closing brackets if needed
        json_text = text.strip()
        # Count opening and closing brackets
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        
        # If there are more opening brackets than closing, add the missing closing brackets
        if open_brackets > close_brackets:
            json_text += ']' * (open_brackets - close_brackets)
        
        # Parse the JSON and format it with indentation
        json_data = json.loads(json_text)
        return json.dumps(json_data, indent=2)
    except json.JSONDecodeError as e:
        # If JSON parsing fails, just return the original content
        return f"Warning: Could not parse as JSON: {e}\n{text}"

# Read notes from CSV
csv_file = 'example_mimic_notes.csv'
clinical_notes = read_notes_from_csv(csv_file)

# Process each note with the model
texts = [PROMPT_TEMPLATE.format(query=note) for note in clinical_notes]
outputs = model.generate(texts, sampling_params)

# Open a file to write results
with open('genie_analysis_results.txt', 'w', encoding='utf-8') as f:
    # Print the results
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        
        # Try to format the response as JSON if possible
        formatted_response = format_json_response(response)
        
        # Print to console
        print(f"\n--- Note {i+1} Analysis ---")
        print(formatted_response)
        print("-" * 50)
        
        # Write to file
        f.write(f"\n--- Note {i+1} Analysis ---\n")
        f.write(formatted_response)
        f.write("\n" + "-" * 50 + "\n")

print(f"\nResults have been saved to 'genie_analysis_results.txt'")
