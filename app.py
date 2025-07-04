from collections import defaultdict
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
import spacy 
import streamlit as st
from textblob import TextBlob

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

def normalise_text(text):
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
    user_input_embeddings = model.encode(user_input, convert_to_tensor=True)
    tags_embeddings = model.encode(tags, convert_to_tensor=True)

    # Compute using cosine similarity (default)
    similarities = util.cos_sim(user_input_embeddings, tags_embeddings)[0]
    matched = [ 
        tag for tag, score in zip(tags, similarities)
        if score >= threshold
    ]
    return matched

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
    normalised = normalise_text(", ".join(user_input))
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
        semantic_matches = embed_user_input_and_tags("".join(corrected), target_list, model)
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
    

st.title("Waku - Career Recommender")

industries = [
    "Medicine / Healthcare", "Law / Legal", "Engineering",
    "Computer Science / IT", "Education", "Business / Finance",
    "Economics", "Arts / Design", "Environment / Sustainability",
    "Psychology / Social Work", "Marketing / Advertising",
    "Government / Public Policy", "Media / Communications",
    "Science / Research", "Trades / Skilled Labor",
    "Entrepreneurship", "Manufacturing", "Logistics / Supply Chain",
    "Hospitality / Tourism", "I'm not sure yet"
]

other_education = [
    "Cert: Google Data Analytics Certificate",            # → Data Scientist
    "Cert: UX Design Professional Certificate",           # → UX Designer
    "Cert: First Aid / CPR",                              # → Registered Nurse
    "Cert: Teaching Certification",                       # → High School Teacher
    "Trade: Electrician Training",                        # → Electrician
    "Cert: Digital Marketing Certificate",                # → Marketing Manager
    "Cert: Culinary Arts Certificate",                    # → Chef
    "Other: Law Prep Program",                            # → Lawyer (minimal alt-ed support)
    "Cert: Human Services Certificate",                   # → Social Worker
    "Cert: Adobe Creative Suite Certification",           # → Graphic Designer
]

alt_ed_topic_to_careers = {
    "Google Data Analytics Certificate": ["Data Scientist"],
    "UX Design Professional Certificate": ["UX Designer"],
    "First Aid / CPR": ["Registered Nurse"],
    "Teaching Certification": ["High School Teacher"],
    "Electrician Training": ["Electrician"],
    "Digital Marketing Certificate": ["Marketing Manager"],
    "Culinary Arts Certificate": ["Chef"],
    "Law Prep Program": ["Lawyer"],
    "Human Services Certificate": ["Social Worker"],
    "Adobe Creative Suite Certification": ["Graphic Designer"]
}


positions = [
    "Software Developer", "UX Designer", "Data Analyst", "IT Support", "Teacher",
    "Marketing Manager", "Chef", "Lawyer", "Social Worker",
    "Sales Associate", "Registered Nurse", "Mechanical Engineer", "High School Teacher",
    "Graphic Designer", "Customer Service Rep", "Freelancer"," Electrician",
    "Other"
]

hard_skills = [
    "Python", "Excel", "CAD", "SQL", 
    "UI/UX Design", "Figma", "User Research", "HTML/CSS",
    "Legal Research", "Litigation", "Contract Law",
    "JavaScript", "Patient Care", "Medical Terminology", "IV Administration",
    "Data Analysis", "Machine Learning", "Mechanical Design", "Accounting",
    "Java", "Lesson Planning", "Subject Expertise", "Classroom Management",
    "Wiring", "Blueprint Reading", "Electrical Code Knowledge", "SEO", 
    "Content Strategy", "Market Analysis", "Culinary Techniques", "Food Safety", 
    "Menu Planning", "Case Management", "Counseling", "Report Writing",
    "Adobe Photoshop", "Typography", "Brand Design",
    "Other"
]

soft_skills = [
    "Leadership", "Communication", "Teamwork", "Problem Solving",
    "Critical Thinking", "Creativity", "Time Management", "Self-Awareness",
    "Adaptability", "Empathy", "Attention to Detail", "Stress Management",
    "Motivated", "Good Listener", "Confidence", "Empathy", 
    "Patience", "Public Speaking", "Manual Dexterity", "Persuasion",
    "Multitasking", "Negotiation", "Written Communication", "Active Listening",
    "Other"
]

# Collect responses from user
user_data = {}

# Basic info
user_data["age"] = st.selectbox("How old are you?", options=[str(i) for i in range(16, 61)], index=None, placeholder="Select your age")
user_data["interested_fields"] = st.multiselect("Are there specific fields you already have an interest in?", options=industries)
user_data["interested_fields_text"] = st.text_input("Other fields of interest (optional):")

# Education info
user_data["in_college"] = st.radio("Have you gone or are you currently in college?", ["Yes", "No"], index=None)
if user_data["in_college"] == "Yes":
    user_data["education_level"] = "bachelor"
    user_data["college_major"] = st.multiselect("What did you study or what are you studying in college?", options=industries, accept_new_options=True)
    user_data["in_postgrad"] = st.radio("Have you done or are you currently pursuing graduate studies?", ["Yes", "No"], index=None)
    if user_data["in_postgrad"] == "Yes":
        user_data["education_level"] = "master+"
        user_data["postgrad_major"] = st.multiselect("What did you study or what are you studying for your postgraduate education?", options=industries, accept_new_options=True)

user_data["alt_education"] = st.multiselect("Have you completed any alternative or non-traditional education?", options=other_education, accept_new_options=True, help="Include certifications, bootcamps, vocational or trade school training, etc.")
parsed_alt_education = [parse_alt_education(e) for e in user_data.get("alt_education", [])] # Parse the user's alt education

# Job info
user_data["previous_experience"] = st.radio("Do you have any previous or current work experience?", ["Yes", "No"], index=None)
if user_data["previous_experience"] == "Yes":
    st.write("Please tell us about your previous or current work experience from past to present. (Optional but recommended):")
    roles = st.multiselect("Fields or roles you've worked in:", options=positions, accept_new_options=True)
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

# Debug print
st.subheader("Collected Input")
st.json(user_data)


# Possible Careers
careers = [
    {
        "title": "Data Scientist",
        "hard_skills": ["Python", "Data Analysis", "Machine Learning"],
        "preferred_fields": ["Tech", "Data", "Finance"],
        "soft_skills": ["Problem Solving", "Critical Thinking", "Communication"],
        "related_majors": ["Computer Science", "Statistics", "Data Science"],
        "required_education": ["bachelor", "master+"],
        "alt_education": ["Bootcamp", "Certificate", "Self-Taught"],
        "experience_level": ["entry", "mid", "senior"],
        "tags": ["Analytical", "Data-Driven", "Curious"]
    },
    {
        "title": "UX Designer",
        "hard_skills": ["UI/UX Design", "Figma", "User Research"],
        "preferred_fields": ["Tech", "Media", "Advertising"],
        "soft_skills": ["Empathy", "Communication", "Adaptability"],
        "related_majors": ["Graphic Design", "Human-Computer Interaction", "Psychology"],
        "required_education": ["bachelor"],
        "alt_education": ["Bootcamp", "Certificate", "Self-Taught"],
        "experience_level": ["entry", "mid"],
        "tags": ["Creative", "User-focused", "Visual"]
    },
    {
        "title": "Registered Nurse",
        "hard_skills": ["Patient Care", "Medical Terminology", "IV Administration"],
        "preferred_fields": ["Healthcare", "Medicine", "Emergency Services"],
        "soft_skills": ["Empathy", "Attention to Detail", "Stress Management"],
        "related_majors": ["Nursing"],
        "required_education": ["associate", "bachelor"],
        "alt_education": ["RN Diploma Program"],
        "experience_level": ["entry", "mid"],
        "tags": ["Compassionate", "Resilient", "Team-Oriented"]
    },
    {
        "title": "High School Teacher",
        "hard_skills": ["Lesson Planning", "Subject Expertise", "Classroom Management"],
        "preferred_fields": ["Education", "Youth Services", "Public Sector"],
        "soft_skills": ["Patience", "Public Speaking", "Empathy"],
        "related_majors": ["Education", "Subject-specific majors (e.g., Math, English)"],
        "required_education": ["bachelor", "master+"],
        "alt_education": ["Teaching Certification", "Alternative Certification Program"],
        "experience_level": ["entry", "mid", "senior"],
        "tags": ["Mentor", "Supportive", "Structured"]
    },
    {
        "title": "Electrician",
        "hard_skills": ["Wiring", "Blueprint Reading", "Electrical Code Knowledge"],
        "preferred_fields": ["Construction", "Maintenance", "Utilities"],
        "soft_skills": ["Problem Solving", "Manual Dexterity", "Attention to Detail"],
        "related_majors": ["Electrical Technology", "Engineering Technology"],
        "required_education": ["high school", "associate"],
        "alt_education": ["Apprenticeship", "Vocational Training"],
        "experience_level": ["entry", "mid", "senior"],
        "tags": ["Hands-on", "Skilled Trade", "Independent"]
    },
    {
        "title": "Marketing Manager",
        "hard_skills": ["SEO", "Content Strategy", "Market Analysis"],
        "preferred_fields": ["Advertising", "Retail", "Tech"],
        "soft_skills": ["Creativity", "Leadership", "Persuasion"],
        "related_majors": ["Marketing", "Business Administration", "Communications"],
        "required_education": ["bachelor", "master+"],
        "alt_education": ["Certificate", "MBA (preferred)"],
        "experience_level": ["mid", "senior"],
        "tags": ["Strategic", "Creative", "Goal-Oriented"]
    },
    {
        "title": "Chef",
        "hard_skills": ["Culinary Techniques", "Food Safety", "Menu Planning"],
        "preferred_fields": ["Hospitality", "Food Services", "Tourism"],
        "soft_skills": ["Creativity", "Multitasking", "Time Management"],
        "related_majors": ["Culinary Arts", "Hospitality Management"],
        "required_education": ["associate"],
        "alt_education": ["Culinary School", "Apprenticeship"],
        "experience_level": ["entry", "mid", "senior"],
        "tags": ["Artistic", "Detail-Oriented", "Energetic"]
    },
    {
        "title": "Lawyer",
        "hard_skills": ["Legal Research", "Litigation", "Contract Law"],
        "preferred_fields": ["Law", "Government", "Business"],
        "soft_skills": ["Critical Thinking", "Negotiation", "Written Communication"],
        "related_majors": ["Law", "Political Science", "Criminal Justice"],
        "required_education": ["master+"],
        "alt_education": [],
        "experience_level": ["mid", "senior"],
        "tags": ["Analytical", "Articulate", "Ethical"]
    },
    {
        "title": "Social Worker",
        "hard_skills": ["Case Management", "Counseling", "Report Writing"],
        "preferred_fields": ["Social Services", "Education", "Healthcare"],
        "soft_skills": ["Empathy", "Active Listening", "Problem Solving"],
        "related_majors": ["Social Work", "Psychology", "Sociology"],
        "required_education": ["bachelor", "master+"],
        "alt_education": ["Human Services Certification"],
        "experience_level": ["entry", "mid"],
        "tags": ["Supportive", "Advocate", "Community-Focused"]
    },
    {
        "title": "Graphic Designer",
        "hard_skills": ["Adobe Photoshop", "Typography", "Brand Design"],
        "preferred_fields": ["Media", "Advertising", "Publishing"],
        "soft_skills": ["Creativity", "Communication", "Time Management"],
        "related_majors": ["Graphic Design", "Visual Arts", "Multimedia Arts"],
        "required_education": ["bachelor"],
        "alt_education": ["Certificate", "Portfolio-Based Entry", "Self-Taught"],
        "experience_level": ["entry", "mid"],
        "tags": ["Visual", "Creative", "Artistic"]
    }
]

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
    user_hard_skills = user_data.get("hard_skills", []) + parse_text_list("hard_skills_text")

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

    st.write("Matched hard skills:", hard_skills_matches)

    # Match soft skills
    user_soft_skills = user_data.get("soft_skills", []) + parse_text_list("soft_skills_text")

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
            normalised = normalise_text(", ".join(topic))
            corrected = correct_spelling(normalised, KNOWN_VOCAB)
            expanded = expand_with_synonyms(corrected, 'n')
            semantic_matches = embed_user_input_and_tags("".join(corrected), career["alt_education"], model, 0.5)

            if expanded in career["alt_education"]:
                score += weights["alt_education"] * 0.7
            elif semantic_matches in career["alt_education"]:
                score += weights["alt_education"] * 0.4

    # Match for tags/personality
    combined_text = f"{user_data.get("likes", "")} {user_data.get("important", "")} {user_data.get("self_description", "")}".lower()

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

    st.write("Matched positive keywords (may not be exact): ", positive_text_matches)

    # Match for negative text
    negative_text = f"{user_data.get("dislikes", "")}".lower()

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
    
    """
    score_breakdown.append({
        "score": score,
        "matches": {
            "hard_skills": hard_skills_matches,
            "soft_skills": soft_skills_matches,
            "industry_match": field_matches,
            "major_match": major_matches,
            "postgrad_match": postgrad_major_matches,
            "alt_education_match": topic,
            "positive_tags": positive_text_matches,
            "negative_tags": negative_text_matches,
            "job_satisfaction_match": job_satisfaction  
        }
    })
    """
    
    print(score_breakdown)

    return score

ranked_careers = sorted(careers, key=lambda c: career_match(user_data, c), reverse=True)

# Display top matches
st.subheader("Top Career Matches:")
for i, career in enumerate(ranked_careers[:3]):
    st.markdown(f"**{i+1}. {career['title']}** — Match Score: {career_match(user_data, career)}")
    