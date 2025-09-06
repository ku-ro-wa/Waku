from collections import defaultdict
from tokenize import String
from nltk.corpus import wordnet as wn
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import json
import os
import pdfplumber
import pytesseract
import spacy 
import streamlit as st

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load sentence-transformers model
@st.cache_resource
def load_st_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_st_model()

# Utility functions
def get_synonyms(word, pos='n'):
    synonyms = set()
    for syn in wn.synsets(word, pos=pos):   # 'n' noun, 'v' verb
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def normalise_and_tokenise_text(text):
    # Join tokens into a string if input is a list
    if isinstance(text, list):
        text = " ".join(text)

    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

def expand_with_synonyms(words, pos='n'):
    expanded = set(words)
    for word in words:
        expanded.update(get_synonyms(word, pos))
    return expanded

def build_known_vocab(careers):
    vocab = set()
    for career in careers:
        vocab.update(map(str.lower, career["hard_skills"]))
        vocab.update(map(str.lower, career["soft_skills"]))
        vocab.update(map(str.lower, career["tags"]))
    return vocab

def correct_spelling(words, known_vocab=None):
    corrected=[]
    for word in words:
        # Add words in KNOWN_VOCAB
        if known_vocab and word.lower() in known_vocab:
            corrected.append(word)
        # Ignore short words
        elif len(word) < 3:
            corrected.append(word)
        else:
            corrected_word = str(TextBlob(word).correct())
            corrected.append(corrected_word)
    return corrected

def parse_alt_education(entry):
    try:
        ed_type, topic = entry.split(":", 1)
        return ed_type.strip(), topic.strip()
    except ValueError:
        return "Other", entry.strip()

def relevant_to_career(topic, career):
    topic_lower = topic.lower()
    fields_to_match = (
        career["hard_skills"]
        + career["related_majors"]
        + career.get("tags", [])
    )
    return any(word.lower() in topic_lower for word in fields_to_match)

def embed_user_input_and_tags(user_input, tags, model, threshold=0.5):
    if isinstance(user_input, list):
        user_input = " ".join(user_input)

    user_input_embeddings = model.encode(user_input, convert_to_tensor=True)
    tags_embeddings = model.encode(tags, convert_to_tensor=True)

    # Check if embeddings are non-empty
    if (
        hasattr(user_input_embeddings, 'shape') and user_input_embeddings.shape[0] > 0
        and hasattr(tags_embeddings, 'shape') and tags_embeddings.shape[0] > 0
    ):
        # Compute using cosine similarity (default)
        similarities = util.cos_sim(user_input_embeddings, tags_embeddings)[0]
        matched = [ 
            tag for tag, score in zip(tags, similarities)
            if score >= threshold
        ]
        return matched
    else:
        return []

# For each match, apply global weight
# Add match into dictionary if not already in dictionary
# If match is in dictionary update weight value if value is higher than current value
def deduplicate_weighted_matches(match_groups: list[tuple[list[str], float]]) -> dict[str, float]:

    weighted_matches = {}    # Dictionary to store each match along with its respective weight
    for matches, weight in match_groups:
        for match in matches:
            weighted_matches[match] = max(weighted_matches.get(match, 0), weight)
    
    return weighted_matches

def match_user_to_targets(
    user_input,
    target_list,
    weight,
    use_synonyms,
    use_semantics,
    pos,
    pos2=None,
    apply_multipliers=False,
    education_required=None,
    user_education=None,
    known_vocab=None,
    model=None
):
    """
    Flexible utility for matching user input (skills, majors, interests, etc.) to job targets.
        
    Parameters:
    - user_input: list of user terms (raw strings)
    - target_list: list of target terms from job profile
    - weight: numeric base weight for the field
    - use_synonyms: whether to expand user input with synonyms
    - use_semantics: whether to perform semantic similarity matching
    - pos: part-of-speech tag for WordNet (e.g., 'n' for nouns)
    - apply_multipliers: apply 1.5x multiplier for education if required level is matched
    - education_required: "bachelor" or "master+"
    - user_level: "bachelor", "master+", etc.
    - known_vocab: set of known valid words for spellchecking
    - model: sentence-transformers model (if use_semantics=True)
    
    Returns:
    - score_contribution (float)
    - matched_items (dict with weights)
    """
    score = 0
    matches = defaultdict(float)

    # Normalise and spell-correct input
    normalised = normalise_and_tokenise_text(user_input)
    corrected = correct_spelling(normalised, known_vocab)

    # Exact matches
    exact_matches = set(target_list).intersection(set([s.strip() for s in corrected]))
    for match in exact_matches:
        matches[match] = max(matches[match], 1.0)

    # Synonym matching
    if use_synonyms:
        expanded = expand_with_synonyms(corrected, pos)
        for match in target_list:
            if match.lower() in expanded:
                matches[match] = max(matches[match], 0.7)

        if pos2 and pos2 != pos:
            expanded2 = expand_with_synonyms(corrected, pos2)
            for match in target_list:
                if match.lower() in expanded2:
                    matches[match] = max(matches[match], 0.5)                    

    # Semantic matching
    if use_semantics and model:
        semantic_matches = embed_user_input_and_tags(corrected, target_list, model)
        for match in semantic_matches:
            matches[match] = max(matches[match], 0.4)

    # Total weighted score
    for match in matches:
        score += weight * matches[match]

    # Education multiplier
    if apply_multipliers and education_required and user_education:
        if user_education == education_required:
            score *= 1.5

    return score, matches

@st.cache_data
def get_data_from_json(filename, required_keys=None):
    """
    Retrieves data from a specified JSON file located in the 'data' directory.

    Parameters:
    - filename (str): The name of the JSON file (e.g., "careers.json").

    Returns:
    - dict or list: The data loaded from the JSON file.
                    Returns None if the file is not found or an error occurs.
    """ 
    # Construct the full path to the JSON file
    data_dir = "data"
    file_path = os.path.join(data_dir, filename)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found at '{file_path}'")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Key validation
    if required_keys and isinstance(data, list):
        for entry in data:
            for key in required_keys:
                if key not in entry:
                    print(f"Warning: Missing key '{key}' in an entry of '{filename}'")

def return_json_list_from_dict(file_name, attribute):
    """
    Converts a dictionary to a list of dictionaries, each containing a single key-value pair.
    
    Returns:
    - list: A list of dictionaries, each with a single key-value pair.
    """
    # Fetch dictionaries from JSON files
    raw_data = get_data_from_json(f"{file_name}.json")
    exists_checker(raw_data)

    # Extract list from dictionary
    list = []
    if raw_data and attribute in raw_data:
        list = raw_data[attribute]
    else:
        st.error(f"Error: '{attribute}' not found in the JSON data.")

    return list

def exists_checker(var):
    """
    Very simple function to check a variable's existence.

    Parameters:
    - var: The name of the variable to be tested.
    
    - Can probably be made more robust.
    Returns:
    - None if the variable exists, or an error message if it does not.
    """
    if var is not None:
        print(f"The variable '{var}' exists.")
    elif var is None:
        st.error(f"Error: the variable '{var}' does not exist/could not be loaded.")

def load_pdf_text(file):
    """Extracts text from a PDF file using pdfplumber, or by using OCR if necessary.

    Parameters:
    - file: The user's uploaded file.

    Returns:
    - str: The extracted text from the PDF.
    """
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            
            # Check if page_text is not None
            if page_text and page_text.strip(): 
                text += page_text + "\n"
            # If page_text is None or empty, use OCR (PDF may be scanned)
            else:
                pil_image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(pil_image)
                text += ocr_text + "\n"

    return text.strip()

def extract_skills(text, skills_list):
    """Extracts skills from a given text using spaCy.
    
    Parameters:
    - text: The input text from which to extract skills.
    - skills_list: A list of skills to look for in the text.

    Returns:
    - list: A list of extracted skills.
    """
    # Join tokens into a string if input is a list
    if isinstance(text, list):
        text = " ".join(text)
    text_lower = text.lower()
    
    matched = []
    # Normalise skills
    for skill in skills_list:
        skill_norm = skill.lower()
        # Check text for multi-word skills
        if skill_norm in text_lower:
            matched.append(skill)
        else:
            # Check individual tokens if single-word skill
            if len(skill_norm.split()) == 1:
                doc = nlp(text_lower)
                if any(token.lemma_ == skill_norm for token in doc if not token.is_stop and token.is_alpha): # Tokenise and lemmatize the text
                    matched.append(skill)
    return matched


def extract_entities(text):
    """Extracts entities from a given text using spaCy."""

    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label == "ORG"]
    dates = [ent.text for ent in doc.ents if ent.label == "DATE"]
    gpes = [ent.text for ent in doc.ents if ent.label == "GPE"]
    langs = [ent.text for ent in doc.ents if ent.label == "LANGUAGE"]

    return orgs, dates, gpes, langs


def parse_resume_to_form(text):
    """Parses the resume text and extracts relevant information for the form."""
    # Join tokens into a string if input is a list
    if isinstance(text, list):
        text = " ".join(text)
    
    # Extract entities from the text
    orgs, dates, gpes, langs = extract_entities(text)
    matched_hard_skills = extract_skills(text, hard_skills)
    matched_soft_skills = extract_skills(text, soft_skills)

    # Create a dictionary to hold the parsed data
    parsed_data = {
        "hard_skills": matched_hard_skills,
        "soft_skills": matched_soft_skills,
        "organizations": orgs,
        "dates": dates,
        "geopolitical_entities": gpes,
        "languages": langs
    }

    return parsed_data

def get_user_skills(data, key="hard_skills"):
    """Extracts skills from user data, handling both list of strings and list of dicts formats.
    
        Parameters:
        - data: The user data dictionary containing skills information.
        - key: The key in the dictionary to look for skills (default is "hard_skills").

        Returns:
        - list: A list of extracted skills.
    """
    if not data:
        return []
    skills = data.get(key, [])
    if f"{key}_text" in data and isinstance(data[f"{key}_text"], str):
        skills += [s.strip() for s in data[f"{key}_text"].split(",") if s.strip()]
    return skills

# Load career data
careers = get_data_from_json("careers.json")
exists_checker(careers)

# Load options data
industries = return_json_list_from_dict("preferred_fields", "preferred_fields")
exists_checker(industries)

majors = return_json_list_from_dict("majors", "related_majors")
exists_checker(majors)

# Currently not used, check alt_edu logic.
alternate_education = return_json_list_from_dict("alternate_education", "alt_education")
exists_checker(alternate_education)

alt_ed_topic_to_careers = get_data_from_json("alt_education_map.json")
exists_checker(alt_ed_topic_to_careers)

career_titles = return_json_list_from_dict("career_titles", "title")
exists_checker(career_titles)

hard_skills = return_json_list_from_dict("hard_skills", "hard_skills")
exists_checker(hard_skills) 

soft_skills = return_json_list_from_dict("soft_skills", "soft_skills")
exists_checker(soft_skills)


# Form title
st.title("Waku - Career Recommender")

# Collect responses from user
user_data = {}

st.write("Welcome to Waku! Let's find your ideal career path together. You can skip any question by leaving it blank, but the more you answer, the better your recommendations will be!")
st.write("You can also upload your CV/resume instead of filling out the form if that's what you prefer. (This feature is not yet implemented)")

uploaded_file = st.file_uploader("Upload your CV/resume (optional)", type=["pdf"], help="This feature is not yet implemented. You can still fill out the form below to get recommendations.")

# Basic info
user_data["age"] = st.selectbox("How old are you?", options=[str(i) for i in range(16, 61)], index=None, placeholder="Select your age")
user_data["interested_fields"] = st.multiselect("Are there specific fields you already have an interest in?", options=industries)
user_data["interested_fields_text"] = st.text_input("Other fields of interest (optional):")

# Education info
user_data["in_college"] = st.radio("Have you gone or are you currently in college?", ["Yes", "No"], index=None)
if user_data["in_college"] == "Yes":
    user_data["education_level"] = "bachelor"
    user_data["college_major"] = st.multiselect("What did you study or what are you studying in college?", options=majors, accept_new_options=True)
    user_data["in_postgrad"] = st.radio("Have you done or are you currently pursuing graduate studies?", ["Yes", "No"], index=None)
    if user_data["in_postgrad"] == "Yes":
        user_data["education_level"] = "master+"
        user_data["postgrad_major"] = st.multiselect("What did you study or what are you studying for your postgraduate education?", options=majors, accept_new_options=True)

user_data["alt_education"] = st.multiselect("Have you completed any alternative or non-traditional education?", options=alt_ed_topic_to_careers, accept_new_options=True, help="Include certifications, bootcamps, vocational or trade school training, etc.")
parsed_alt_education = [parse_alt_education(e) for e in user_data.get("alt_education", [])] # Parse the user's alt education

# Job info
user_data["previous_experience"] = st.radio("Do you have any previous or current work experience?", ["Yes", "No"], index=None)
if user_data["previous_experience"] == "Yes":
    st.write("Please tell us about your previous or current work experience from past to present. (Optional but recommended):")
    roles = st.multiselect("Fields or roles you've worked in:", options=career_titles, accept_new_options=True)
    experience_list = []

    for role in roles:
        yoe = st.slider(f"Years of experience as {role}", 0, 20, 1, key=f"yoe_{role}")
        experience_list.append({"role": role, "yoe": yoe})

    user_data["experience_details"] = experience_list
    user_data["job_satisfaction"] = st.slider("On a scale of 1-10 how satisfied are you with your current job?", 0, 10, 0, 1)

# Skills
user_data["hard_skills"] = st.multiselect("Select your hard skills:", options=hard_skills)
user_data["hard_skills_text"] = st.text_input("Other hard skills (optional):")
user_data["soft_skills"] = st.multiselect("Select your soft skills:", options=soft_skills)
user_data["soft_skills_text"] = st.text_input("Other soft skills (optional):")

# Qualitative inputs
user_data["likes"] = st.text_area("What kinds of topics, environments, or tasks do you enjoy?")
user_data["dislikes"] = st.text_area("Are there types of work or settings you'd prefer to avoid?")
user_data["important"] = st.text_area("What is important to you in a career? (You can answer in a few words or sentences)")
user_data["self_description"] = st.text_area("Describe yourself in a few words or sentences!")

# Debug print (temp)
st.subheader("Collected Input")
st.json(user_data)

# Preparing the PDF text and extracting skills
pdf_file = load_pdf_text(uploaded_file) if uploaded_file else None
normalised_pdf_text = normalise_and_tokenise_text(pdf_file) if pdf_file else None

# debug print
print("Extracted PDF text:", normalised_pdf_text)

matched_hard_skills = extract_skills(normalised_pdf_text, hard_skills) if normalised_pdf_text else None
matched_soft_skills = extract_skills(normalised_pdf_text, soft_skills) if normalised_pdf_text else None

# debug prints
print("Matched hard skills from PDF:", matched_hard_skills)
print("Matched soft skills from PDF:", matched_soft_skills)

parsed_data = parse_resume_to_form(normalised_pdf_text) if normalised_pdf_text else None

# temp debug
print("Parsed PDF data:", parsed_data)

# Initialise list for feedback
score_breakdown = []

# Initialise vocab
KNOWN_VOCAB = build_known_vocab(careers)

# Basic matching logic
def career_match(user_data, career):
    score = 0
    weights = {
        "hard_skills": 2,
        "soft_skills": 1.5,
        "fields": 2,
        "education": 2,
        "alt_education": 1,
        "tags": 1
    }

    def parse_text_list(field):
        value = user_data.get(field, "")
        if not isinstance(value, str):
            return []
        return [s.strip() for s in value.split(",") if s.strip()]

    # Match hard skills 
    user_hard_skills = get_user_skills(user_data if not uploaded_file else parsed_data, key="hard_skills")

    hs_add_to_score, hard_skills_matches = match_user_to_targets(
        user_input=user_hard_skills,
        target_list=career["hard_skills"],
        weight=weights["hard_skills"],
        use_synonyms=True,
        use_semantics=False,
        pos='n',
        known_vocab=KNOWN_VOCAB
    )

    score += hs_add_to_score

    if hard_skills_matches:
        st.write("Matched hard skills:", hard_skills_matches)

    # Match soft skills
    user_soft_skills = get_user_skills(user_data if not uploaded_file else parsed_data, key="soft_skills")

    ss_add_to_score, soft_skills_matches = match_user_to_targets( 
        user_input=user_soft_skills,
        target_list=career["soft_skills"],
        weight=weights["soft_skills"],
        use_synonyms=True,
        use_semantics=False,
        pos='n',
        known_vocab=KNOWN_VOCAB
    )

    score += ss_add_to_score

    if soft_skills_matches:
        st.write("Matches soft skills: ", soft_skills_matches)

    # Match fields/industries
    user_fields = user_data.get("interested_fields", []) + parse_text_list("intereseted_fields_text")

    fields_add_to_score, field_matches = match_user_to_targets(
        user_input=user_fields,
        target_list=career["preferred_fields"],
        weight=weights["fields"],
        use_synonyms=True,
        use_semantics=True,
        pos='n',
        known_vocab=KNOWN_VOCAB,
        model=model
    )

    score += fields_add_to_score

    if field_matches:
        st.write("Matched fields/industries: ", field_matches)

    # Match college majors
    college_majors = user_data.get("college_major", [])

    majors_add_to_score, major_matches = match_user_to_targets(
        user_input=college_majors,
        target_list=career["related_majors"],
        weight=weights["education"],
        use_synonyms=True,
        use_semantics=True,
        pos='n',
        apply_multipliers=True,
        education_required='bachelor',
        user_education='bachelor',
        known_vocab=KNOWN_VOCAB,
        model=model
    )

    score += majors_add_to_score

    if major_matches:
        st.write("Matched majors: ", major_matches)

    # Match postgrad majors
    postgrad_majors = user_data.get("postgrad_major", [])

    postgrad_majors_add_to_score, postgrad_major_matches = match_user_to_targets(
        user_input=postgrad_majors,
        target_list=career["related_majors"],
        weight=weights["education"],
        use_synonyms=True,
        use_semantics=True,
        pos='n',
        apply_multipliers=True,
        education_required='master+',
        user_education='master+',
        known_vocab=KNOWN_VOCAB,
        model=model
    )

    score += postgrad_majors_add_to_score

    if postgrad_major_matches:
        st.write("Matched postgrad majors: ", postgrad_major_matches)

    # Match alt education 
    for ed_type, topic in parsed_alt_education:
        if ed_type in career["alt_education"]:
            score += weights["alt_education"] * 0.4
        
        if career["title"] in alt_ed_topic_to_careers.get(topic, []):
            score += weights["alt_education"] * 1

        elif relevant_to_career(topic, career):
            score += weights["alt_education"] * 0.7

        else:
            normalised = normalise_and_tokenise_text(topic)
            corrected = correct_spelling(normalised, KNOWN_VOCAB)
            expanded = expand_with_synonyms(corrected, 'n')
            semantic_matches = embed_user_input_and_tags(corrected, career["alt_education"], model, 0.5)

            if expanded in career["alt_education"]:
                score += weights["alt_education"] * 0.7
            elif semantic_matches in career["alt_education"]:
                score += weights["alt_education"] * 0.4

    # Match for tags/personality
    combined_text = f"{user_data.get('likes', '')} {user_data.get('important', '')} {user_data.get('self_description', '')}".lower()

    positive_text_add_to_score, positive_text_matches = match_user_to_targets(
        user_input=combined_text,
        target_list=career["tags"],
        weight=weights["tags"],
        use_synonyms=True,
        use_semantics=True,
        pos='n',
        pos2='v',
        known_vocab=KNOWN_VOCAB,
        model=model
    ) 

    score += positive_text_add_to_score

    if positive_text_matches:
        st.write("Matched positive keywords (may not be exact): ", positive_text_matches)

    # Match for negative text
    negative_text = f"{user_data.get('dislikes', '')}".lower()

    negative_text_minus_from_score, negative_text_matches = match_user_to_targets(
        user_input=negative_text,
        target_list=career["tags"],
        weight=weights["tags"],
        use_synonyms=True,
        use_semantics=True,
        pos='v',
        known_vocab=KNOWN_VOCAB,
        model=model
    )

    score -= negative_text_minus_from_score

    if negative_text_matches:
        st.write("Matched negative keywords (may not be exact): ", negative_text_matches)


    # Job Satisfaction match
    job_satisfaction = user_data.get("job_satisfaction", 5)
    job_satisfaction = int(job_satisfaction)
    roles = [experience["role"] for experience in user_data.get("experience_details", [])]

    # If career matches the user's current job (Assuming they list their jobs from past to present).
    if roles:
        if career["title"] == roles[-1] and job_satisfaction > 5:
            score += 1
        elif career["title"] == roles[-1] and job_satisfaction < 5:
            score -= 1
        else:
            pass
    
    # Experience match
    for exp in user_data.get("experience_details", []):
        if exp["role"] == career["title"]:
            level_weights = {"entry": 0.4, "mid": 0.6, "senior": 0.8}
            levels = career.get("experience_level", [])
            multiplier = max([level_weights.get(level, 0) for level in levels], default=0.5)
            score += min(int(exp["yoe"]), 15) * multiplier
    

    
    print(score_breakdown)

    return score


# Display career matches
if not uploaded_file:
    ranked_careers = sorted(careers, key=lambda c: career_match(user_data, c), reverse=True) 
elif uploaded_file:
    ranked_careers = sorted(careers, key=lambda c: career_match(parsed_data, c), reverse=True)

# Display top matches (form)
st.subheader("Top Career Matches:")
for i, career in enumerate(ranked_careers[:3]):
    if not uploaded_file:
        st.markdown(f"**{i+1}. {career['title']}** — Match Score: {career_match(user_data, career)}")
    elif uploaded_file:
        st.markdown(f"**{i+1}. {career['title']}** — Match Score: {career_match(parsed_data, career)}")

