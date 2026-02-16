from collections import defaultdict
from unittest import result
from dotenv import load_dotenv
from tokenize import String
from nltk.corpus import wordnet as wn
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import json
import os
import pdfplumber
import pytesseract
import spacy 
import streamlit as st
import time

# Import the logger
from src.logger import get_session_logger

load_dotenv()  # Load environment variables from .env file

client = OpenAI()

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load sentence-transformers model
@st.cache_resource
def load_st_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_st_model()

# Utility functions

# Get synonyms from WordNet
def get_synonyms(word, pos='n'):
    synonyms = set()
    
    for syn in wn.synsets(word, pos=pos):   # 'n' noun, 'v' verb
        # Should never be None, just for Pylance
        if syn is None:
            continue

        lemmas = syn.lemmas()
        if lemmas:
            for lemma in lemmas:
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != word:
                    synonyms.add(synonym)
        
    return list(synonyms)

# Normalise and tokenise text using spaCy
def normalise_and_tokenise_text(text):
    # Join tokens into a string if input is a list
    if isinstance(text, list):
        text = " ".join(text)

    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

# Expand words with synonyms
def expand_with_synonyms(words, pos='n'):
    expanded = set(words)
    for word in words:
        expanded.update(get_synonyms(word, pos))
    return expanded

# Build known vocabulary from career data for spellchecking
def build_known_vocab(careers):
    vocab = set()
    for career in careers:
        vocab.update(map(str.lower, career["hard_skills"]))
        vocab.update(map(str.lower, career["soft_skills"]))
        vocab.update(map(str.lower, career["tags"]))
    return vocab

# Correct spelling using TextBlob
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

# Parse user-inputted alternate education entries
def parse_alt_education(entry):
    try:
        ed_type, topic = entry.split(":", 1)
        return ed_type.strip(), topic.strip()
    except ValueError:
        return "Other", entry.strip()

# Check if alt education topic is relevant to career
def relevant_to_career(topic, career):
    topic_lower = topic.lower()
    fields_to_match = (
        career["hard_skills"]
        + career["related_majors"]
        + career.get("tags", [])
    )
    return any(word.lower() in topic_lower for word in fields_to_match)

# Embed user input and tags, return matches above threshold
def embed_user_input_and_tags(user_input, tags, model, threshold=0.5):
    if isinstance(user_input, list):
        user_input = " ".join(user_input)
    
    # Return early if input is empty to avoid false semantic matches
    if not user_input or not user_input.strip():
        return []

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

# Main matching function, see details below
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
    - dict or list: The data loaded from the JSON file. Returns None if the file is not found or an error occurs.
    """ 
    # Construct the full path to the JSON file
    data_dir = "data"
    file_path = os.path.join(data_dir, filename)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Key validation
        if required_keys and isinstance(data, list):
            for entry in data:
                for key in required_keys:
                    if key not in entry:
                        print(f"Warning: Missing key '{key}' in an entry of '{filename}'")
        
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


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
    """Extracts skills from a given text using spaCy with improved matching.
    
    Parameters:
    - text: The input text from which to extract skills.
    - skills_list: A list of skills to look for in the text.

    Returns:
    - list: A list of extracted skills (deduplicated).
    """
    # Join tokens into a string if input is a list
    if isinstance(text, list):
        text = " ".join(text)
    text_lower = text.lower()
    
    matched = set()  # Use set to avoid duplicates
    doc = nlp(text_lower)  # Parse once for efficiency
    text_lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    text_lemmas_set = set(text_lemmas)
    
    # Build text with lemmatized version for better matching
    text_lemmatized = " ".join(text_lemmas)
    
    # Normalise skills
    for skill in skills_list:
        skill_lower = skill.lower()
        
        # Exact substring match (for multi-word skills)
        if skill_lower in text_lower:
            matched.add(skill)
            continue
            
        # Lemmatized matching for multi-word skills
        skill_lemmas = [token.lemma_ for token in nlp(skill_lower) if not token.is_stop and token.is_alpha]
        skill_lemmatized = " ".join(skill_lemmas)
        if skill_lemmatized in text_lemmatized and skill_lemmatized:  # Avoid empty lemma strings
            matched.add(skill)
            continue
        
        # For single-word skills, check individual tokens
        if len(skill_lower.split()) == 1:
            skill_lemma = nlp(skill_lower)[0].lemma_ if nlp(skill_lower) else skill_lower
            if skill_lemma in text_lemmas_set:
                matched.add(skill)
    
    return list(matched)


def extract_entities(text):
    """Extracts entities from a given text using spaCy.
    
    Parameters:
    - text: The input text from which to extract entities.

    Returns:
    - tuple: A tuple containing lists of organizations, dates, geopolitical entities, and languages.
    """

    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    langs = [ent.text for ent in doc.ents if ent.label_ == "LANGUAGE"]

    # Supplement with keyword search for languages
    language_keywords = {"french", "dutch", "english", "spanish", "german", "italian", "portuguese", "russian", "chinese", "japanese", "arabic", "hindi"}
    found_langs = [word.capitalize() for word in language_keywords if word in text.lower()]
    langs = list(set(langs + found_langs))

    return orgs, dates, gpes, langs


def parse_resume_to_form(text, use_llm=True):
    """Parses resume text into structured data with optional LLM enhancement.
    
    Parameters:
    - text: The resume text to be parsed.
    - use_llm: Whether to use LLM for better context extraction (default True).

    Returns:
    - dict: A dictionary containing extracted hard skills, soft skills, organizations, dates, 
            geopolitical entities, languages, and LLM-extracted insights.
    """
    if isinstance(text, list):
        text = " ".join(text)

    orgs, dates, gpes, langs = extract_entities(text)
    matched_hard_skills = extract_skills(text, hard_skills)
    matched_soft_skills = extract_skills(text, soft_skills)

    parsed_data = {
        "hard_skills": matched_hard_skills,
        "soft_skills": matched_soft_skills,
        "organizations": orgs,
        "dates": dates,
        "geopolitical_entities": gpes,
        "languages": langs,
        "raw_text": text  # Store raw text for semantic matching
    }
    
    # Use LLM to extract additional context (certifications, achievements, etc.)
    if use_llm:
        try:
            llm_insights = extract_resume_context_via_llm(text)
            if llm_insights:
                parsed_data.update(llm_insights)
        except Exception as e:
            print(f"LLM context extraction failed: {e}")
            # Gracefully degrade - continue with base extraction
    
    return parsed_data


def extract_resume_context_via_llm(text, max_tokens=300):
    """Extracts additional context from resume using LLM (certifications, achievements, etc.).
    
    Parameters:
    - text: The resume text to analyze.
    - max_tokens: Maximum tokens for LLM response.

    Returns:
    - dict: Dictionary with extracted context including certifications, key_achievements, and career_interests.
    """
    sys_instruction = (
        "Extract structured information from this resume. Return JSON with: "
        "1. 'certifications': list of courses, bootcamps, certifications mentioned "
        "2. 'key_achievements': list of 3-4 major accomplishments or projects "
        "3. 'career_context': 1-2 sentences about the person's career trajectory and interests. "
        "Be precise and extract only information explicitly stated."
    )
    
    text_input = f"Resume: {text[:3000]}"  # Limit text length to control costs
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_instruction},
                {"role": "user", "content": text_input},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.3,  # Lower temp for factual extraction
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        
        content = response.choices[0].message.content
        if content:
            extracted = json.loads(content)
            return {
                "certifications": extracted.get("certifications", []),
                "key_achievements": extracted.get("key_achievements", []),
                "career_context": extracted.get("career_context", "")
            }
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return None
    
    return None



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


def format_user_data_to_text(user_data):
    """Converts structured form data into a coherent text block.
    
    Parameters:
    user_data: The user's form input data dictionary.

    Returns:
    - str: A formatted text representation of the user's data.
    """
    parts = []
    if user_data.get('interested_fields'):
        parts.append(f"Interested fields: {', '.join(user_data['interested_fields'])}")
    if user_data.get('education_level'):
        parts.append(f"Education level: {user_data['education_level']}")
        if user_data.get('college_major'):
            parts.append(f"College major: {user_data['college_major']}")
        if user_data.get('postgrad_major'):
            parts.append(f"Postgraduate major: {user_data['postgrad_major']}")
    if user_data.get('alt_education'):
        parts.append(f"Alternative education: {', '.join(user_data['alt_education'])}")
    if user_data.get('experience_details'):
        exp_details = "; ".join([f"{exp['role']} ({exp['yoe']} years)" for exp in user_data['experience_details']])
        parts.append(f"Work experience: {exp_details}")
        if user_data.get('job_satisfaction') is not None:
            parts.append(f"Job satisfaction: {user_data['job_satisfaction']}/10")
    if user_data.get('hard_skills'):
        parts.append(f"Hard skills: {', '.join(user_data['hard_skills'])}")
    if user_data.get('soft_skills'):
        parts.append(f"Soft skills: {', '.join(user_data['soft_skills'])}")
    if user_data.get('likes'):
        parts.append(f"Likes: {user_data['likes']}")
    if user_data.get('dislikes'):
        parts.append(f"Dislikes: {user_data['dislikes']}")
    if user_data.get('important'):
        parts.append(f"Important in a career: {user_data['important']}")
    if user_data.get('self_description'):
        parts.append(f"Self-description: {user_data['self_description']}")
    
    return " ".join(parts)


def get_user_input_summary(text, uploaded_file=False, use_responses_api=True, temperature=0.7, max_output_tokens=400):
    """Generates a summary of the user input text using OpenAI's API.
    
    Parameters:
    - text: The user input text (resume text string or form data dict) to be summarized.
    - uploaded_file: Whether the text is from an uploaded file (default is False).
    - use_responses_api: Whether to use the new Responses API (default is True).
    - temperature: The temperature setting for the classic Chat API (default is 0.7).
    - max_output_tokens: The maximum number of tokens for the output summary (default is 400).

    Returns:
    - str: The generated summary of the user input.
    """

    if uploaded_file == True:
        sys_instruction = (
            "Summarize the following résumé in a short, engaging paragraph that directly addresses the user as 'you'."
            "Briefly highlight key skills, education, and experience."
            "Provide suggestions for suitable career paths based on the résumé content."
            "Add a light, witty tone without exaggeration or filler."
        )
        text_input = f"Resume text: {text}"
    else:
        sys_instruction = (
        "Create an engaging summary based on the user's form responses."
        "Address them as 'you' and highlight the unique aspects and combinations of their skills, education, and experiences."
        "Provide suggestions for suitable career paths based on the provided information."
        "Maintain a light, witty tone without exaggeration or filler."
    )
        text_input = format_user_data_to_text(text)


    for attempt in range(3):  # Retry mechanism
        if use_responses_api:
            # New API (does NOT support temperature)
            response = client.responses.create(
                model="gpt-4o-mini",
                instructions=sys_instruction,
                input=text_input,
                max_output_tokens=max_output_tokens,
            )

            if response.status == "incomplete":
                reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
                print(f"Response incomplete (Reason: {reason}).")
                max_output_tokens *= 2
                time.sleep(0.5)  
                continue

            if hasattr(response, 'output_text') and response.output_text:
                return response.output_text.strip()
            elif hasattr(response, 'output') and response.output and len(response.output) > 0:
                output_item = response.output[0]
                content = getattr(output_item, 'content', None)
                if content and isinstance(content, str):
                    return content.strip()
            return "No summary was generated."

        else:
            # Classic Chat API (DOES support temperature)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_instruction},
                    {"role": "user", "content": text_input},
                ],
                max_completion_tokens=max_output_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            return "No summary was generated."

    return "Failed to generate summary after multiple attempts."


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

# Initialise session logger
logger = get_session_logger()
logger.log_event("session_started", {"timestamp": str(time.time())})



# Streamlit app layout
st.title("Waku - Career Recommender")

# Collect responses from user
user_data = {}

st.write("Welcome to Waku! Let's find your ideal career path together. You can skip any question by leaving it blank, but the more you answer, the better your recommendations will be!")
st.write("You can also upload your CV/resume instead of filling out the form if that's what you prefer. (Your data will be utilised by OpenAI's API for analysis, but won't be stored.)")

uploaded_file = st.file_uploader("Upload your CV/resume (optional)", type=["pdf"])

# Basic info
user_data["interested_fields"] = st.multiselect("Are there specific fields you already have an interest in?", options=industries)
user_data["interested_fields_text"] = st.text_input("Other fields of interest (optional):")

# Education info
user_data["in_college"] = st.radio("Have you gone or are you currently in college?", ["Yes", "No"], index=None)
if user_data["in_college"] == "Yes":
    user_data["education_level"] = "bachelor"
    user_data["college_major"] = st.multiselect("What did you study or what are you studying in college?", options=majors or [], accept_new_options=True)
    user_data["in_postgrad"] = st.radio("Have you done or are you currently pursuing graduate studies?", ["Yes", "No"], index=None)
    if user_data["in_postgrad"] == "Yes":
        user_data["education_level"] = "master+"
        user_data["postgrad_major"] = st.multiselect("What did you study or what are you studying for your postgraduate education?", options=majors or [], accept_new_options=True)

user_data["alt_education"] = st.multiselect("Have you completed any alternative or non-traditional education?", options=alt_ed_topic_to_careers or {}, accept_new_options=True, help="Include certifications, bootcamps, vocational or trade school training, etc.")
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

# Confirmation checkbox
user_data["confirm"] = st.checkbox("I want the info inputted in this form to be used for career recommendations (You can still edit your inputs after confirming).")

# Log all user input data
logger.log_all_user_data(user_data)

# Preparing the PDF text and extracting skills
pdf_file = load_pdf_text(uploaded_file) if uploaded_file else None

# Log PDF upload
if uploaded_file:
    logger.log_pdf_upload(uploaded_file.name, len(pdf_file) if pdf_file else 0)

# Parse resume BEFORE normalising (preserve structure for better extraction)
if pdf_file:
    parsed_data = parse_resume_to_form(pdf_file, use_llm=True)
    logger.log_parsed_data("resume", parsed_data)
    print("Parsed PDF data:", parsed_data)
else:
    parsed_data = None

# For semantic matching and qualitative analysis, keep normalised text
normalised_pdf_text = normalise_and_tokenise_text(pdf_file) if pdf_file else None
print("Extracted PDF text (normalised):", normalised_pdf_text)

# Initialise list for feedback
score_breakdown = []

# Initialise vocab
KNOWN_VOCAB = build_known_vocab(careers)


# Basic matching logic
def career_match(user_data, career):
    """Calculates a match score between user data and a career.
    
    Parameters: 
    - user_data: The user's input data (from form or parsed resume).
    - career: The career profile to match against.

    Returns:
    - float: The calculated match score.
    """
    score = 0   # Initialise to 0
    weights = {
        "hard_skills": 2,
        "soft_skills": 1.5,
        "fields": 2,
        "education": 2,
        "alt_education": 1,
        "tags": 1
    }

    # Helper to parse comma-separated text inputs
    def parse_text_list(field):
        value = user_data.get(field, "")
        if not isinstance(value, str):
            return []
        return [s.strip() for s in value.split(",") if s.strip()]

    # Match hard skills (use semantic matching for PDFs for better coverage)
    user_hard_skills = get_user_skills(user_data if not uploaded_file else parsed_data, key="hard_skills")
    use_semantics_for_skills = uploaded_file  # Enable semantics only for resume PDFs

    hs_add_to_score, hard_skills_matches = match_user_to_targets(
        user_input=user_hard_skills,
        target_list=career["hard_skills"],
        weight=weights["hard_skills"],
        use_synonyms=True,
        use_semantics=use_semantics_for_skills,  # Use semantics for PDF resumes
        pos='n',
        known_vocab=KNOWN_VOCAB,
        model=model if use_semantics_for_skills else None
    )

    score += hs_add_to_score

    #if hard_skills_matches:
    #   print("Matched hard skills:", hard_skills_matches)

    # Match soft skills (use semantic matching for PDFs)
    user_soft_skills = get_user_skills(user_data if not uploaded_file else parsed_data, key="soft_skills")
    use_semantics_for_soft = uploaded_file  # Enable semantics for resume PDFs

    ss_add_to_score, soft_skills_matches = match_user_to_targets( 
        user_input=user_soft_skills,
        target_list=career["soft_skills"],
        weight=weights["soft_skills"],
        use_synonyms=True,
        use_semantics=use_semantics_for_soft,  # Use semantics for PDF resumes
        pos='n',
        known_vocab=KNOWN_VOCAB,
        model=model if use_semantics_for_soft else None
    )

    score += ss_add_to_score

    #if soft_skills_matches:
    #    print("Matched soft skills: ", soft_skills_matches)

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

    #if field_matches:
    #    print("Matched fields/industries: ", field_matches)

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

    #if major_matches:
    #    print("Matched majors: ", major_matches)

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

    #if postgrad_major_matches:
    #    print("Matched postgrad majors: ", postgrad_major_matches)

    # Match alt education 
    for ed_type, topic in parsed_alt_education:
        if ed_type in career["alt_education"]:
            score += weights["alt_education"] * 0.4
        
        if alt_ed_topic_to_careers and career["title"] in alt_ed_topic_to_careers.get(topic, []):
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
    if uploaded_file and parsed_data:
        # For PDFs, use LLM-extracted context + achievements + raw text
        combined_text = f"{' '.join(parsed_data.get('key_achievements', []))} {parsed_data.get('career_context', '')} {parsed_data.get('raw_text', '')}".lower()
    else:
        # For form inputs, use user's self-description
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

    #if positive_text_matches:
    #    print("Matched positive keywords (may not be exact): ", positive_text_matches)

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

    #if negative_text_matches:
    #    print("Matched negative keywords (may not be exact): ", negative_text_matches)


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


# Display top matches (CV upload)
ranked_careers = []
if careers:
    if not uploaded_file and user_data["confirm"]:
        ranked_careers = sorted(careers, key=lambda c: career_match(user_data, c), reverse=True) 
    elif uploaded_file and parsed_data:
        ranked_careers = sorted(careers, key=lambda c: career_match(parsed_data, c), reverse=True)

# Log career matches
logger.log_career_matches(ranked_careers, top_n=10)

# Display top matches (form)
st.subheader("Your Top Career Matches:")
st.write("Disclaimer: Match scores are only indicative of how much your inputted data aligns with our career profiles. Hence why resume uploads may yield lower scores due to less structured data.")
for i, career in enumerate(ranked_careers[:3]):
    if (not uploaded_file and career_match(user_data, career) < 1) or (uploaded_file and career_match(parsed_data, career) < 1):
        st.markdown("No matches yet! Please fill in the form a bit more or upload a valid resume for Waku Bot™'s insights!")
        break
    elif not uploaded_file and career_match(user_data, career) >= 1 and user_data["confirm"]:
        st.markdown(f"**{i+1}. {career['title']}** — Match Score: {career_match(user_data, career):.1f}")
    elif uploaded_file and career_match(parsed_data, career) >= 1:
        st.markdown(f"**{i+1}. {career['title']}** — Match Score: {career_match(parsed_data, career):.1f}")
    
# Display resume summary (if uploaded)
st.subheader("Waku Bot™ Summary of Your Resume:")
with st.spinner("Generating summary..."):
    summary = ""
    if uploaded_file and pdf_file:
        summary = get_user_input_summary(pdf_file, uploaded_file=True)
        logger.log_summary_generated(summary, len(summary) if summary else 0)
    elif not uploaded_file and user_data.get("confirm", True):
        summary = get_user_input_summary(user_data, uploaded_file=False)
        logger.log_summary_generated(summary, len(summary) if summary else 0)
            
    if summary == "" or "Response incomplete" in summary:
        st.write("No generated summary yet! Please ensure that a valid resume has been uploaded!")
    else:
        st.write(summary)

# Debug print (temp)
# st.subheader("Collected Input")
# st.json(user_data)

# Finalize session log
logger.finalise()

# Display log file location and download options
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Session Logs")

try:
    # Read and display the log file content
    with open(logger.log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Display in sidebar
    with st.sidebar.expander("View Session Log (.log)"):
        st.text(log_content)
    
    # Download button for .log file
    st.sidebar.download_button(
        label="⬇️ Download Log (.log)",
        data=log_content,
        file_name=f"session_{logger.session_id}.log",
        mime="text/plain"
    )
except Exception as e:
    st.sidebar.warning(f"Could not read log file: {e}")

try:
    # Read and display the JSON log file
    with open(logger.json_log_file, 'r', encoding='utf-8') as f:
        json_content = f.read()
    
    # Display in sidebar
    with st.sidebar.expander("View Session Log (.json)"):
        st.json(json.loads(json_content))
    
    # Download button for .json file
    st.sidebar.download_button(
        label="⬇️ Download Log (.json)",
        data=json_content,
        file_name=f"session_{logger.session_id}.json",
        mime="application/json"
    )
except Exception as e:
    st.sidebar.warning(f"Could not read JSON log file: {e}")

st.sidebar.info(f"Session ID: `{logger.session_id}`")

