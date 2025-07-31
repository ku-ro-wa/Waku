import os
import json

careers = []

data_dir = "data"
file_path = os.path.join(data_dir, "careers.json")

os.makedirs(data_dir, exist_ok=True)

# Extract careers from JSON file
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        careers = json.load(f)
except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{file_path}'. Check file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


def read_attributes(careers, attribute, output_file):
    """Read specific attributes from a career dictionary."""
    # Set for unique values
    unique_values = set()
    
    # Get the attribute from each career
    for career in careers:
        value = career.get(attribute)

        # Skip if the value is None
        if value is None:
            continue

        # Check if the value is a list or a string
        # If it's a list, add each item; if it's a string, add it directly    
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    unique_values.add(item)
        elif isinstance(value, str):
            unique_values.add(value)
        # If the value is of an unexpected type, print a warning
        else:
            print(f"Warning: Unexpected type for attribute '{attribute}' in career: {career}")
    
    # Convert the set to a sorted list
    unique_values_list = sorted(list(unique_values))

    # Format the output data
    output_data = {attribute: unique_values_list}

    output_file_path = os.path.join(data_dir, f"{output_file}.json")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error: Could not write to '{output_file_path}'. {e}")


# Function calls to read attributes and write to JSON files
read_attributes(careers, "title", "career_titles")
read_attributes(careers, "category", "categories")
read_attributes(careers, "description", "descriptions")
read_attributes(careers, "hard_skills", "hard_skills")
read_attributes(careers, "preferred_fields", "preferred_fields")
read_attributes(careers, "soft_skills", "soft_skills")
read_attributes(careers, "related_majors", "majors")
read_attributes(careers, "alt_education", "alternate_education")
read_attributes(careers, "tags", "tags")