import os
import json

careers = []

data_dir = "data"
file_path = os.path.join(data_dir, "careers.json")

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

# Write extracted data to JSON files
career_titles = [{"title": career["title"]} for career in careers]
with open(os.path.join(data_dir, "career_titles.json"), 'w', encoding='utf-8') as f:
    json.dump(career_titles, f, indent=2)

industries = [{"preferred_fields": career["preferred_fields"]} for career in careers]
with open(os.path.join(data_dir, "industries.json"), 'w', encoding='utf-8') as f:
    json.dump(industries, f, indent=2)

hard_skills = [{"hard_skills": career["hard_skills"]} for career in careers]
with open(os.path.join(data_dir, "hard_skills.json"), 'w', encoding='utf-8') as f:
    json.dump(hard_skills, f, indent=2)

soft_skills = [{"soft_skills": career["soft_skills"]} for career in careers]
with open(os.path.join(data_dir, "soft_skills.json"), 'w', encoding='utf-8') as f:
    json.dump(soft_skills, f, indent=2)

majors = [{"related_majors": career["related_majors"]} for career in careers]
with open(os.path.join(data_dir, "majors.json"), 'w', encoding='utf-8') as f:
    json.dump(majors, f, indent=2)

alternate_education = [{"alt_education": career["alt_education"]} for career in careers]
with open(os.path.join(data_dir, "alternate_education.json"), 'w', encoding='utf-8') as f:
    json.dump(alternate_education, f, indent=2)

tags = [{"tags": career["tags"]} for career in careers]
with open(os.path.join(data_dir, "tags.json"), 'w', encoding='utf-8') as f:
    json.dump(tags, f, indent=2)
