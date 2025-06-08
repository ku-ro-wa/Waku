import spacy 
from nltk.corpus import wordnet as wn
import streamlit as st

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Utility functions
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_"," ").lower()
            if synonym != word:
                synonym.add(synonym)
    return list(synonyms)

def normalise_text(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

def expand_with_synonyms(words):
    expanded = set(words)
    for word in words:
        expanded.update(get_synonyms(word))
    return expanded



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
    "Cert: AWS Certified Solutions Architect",
    "Cert: Google Data Analytics Certificate",
    "Cert: PMP – Project Management Professional",
    "Cert: CompTIA Security+",
    "Cert: First Aid / CPR",
    "Bootcamp: Full-Stack Web Development",
    "Bootcamp: Data Science Immersive",
    "Bootcamp: Cybersecurity",
    "Bootcamp: UI/UX Design",
    "Trade: Electrician Training",
    "Trade: Welding Certification",
    "Trade: Automotive Technician Training",
    "Vocational: CNA Certification",
    "Vocational: Real Estate License",
    "Vocational: Massage Therapy Certification",
    "Other: Coursera / edX Certificates",
    "Other: Udacity Nanodegree",
    "Other: Google Career Certificates"
]

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
user_data["interested_fields"] = st.multiselect("Are there specific industries you already have an interest in?", options=industries, accept_new_options=True)

# Education info
user_data["in_college"] = st.radio("Have you gone or are you currently in college?", ["Yes", "No"], index=None)
if user_data["in_college"] == "Yes":
    user_data["college_major"] = st.multiselect("What did you study or what are you studying in college?", options=industries, accept_new_options=True)
    user_data["in_postgrad"] = st.radio("Have you done or are you currently pursuing graduate studies?", ["Yes", "No"], index=None)
    if user_data["in_postgrad"] == "Yes":
        user_data["postgrad_major"] = st.multiselect("What did you study or what are you studying for your postgraduate education?", options=industries, accept_new_options=True)

user_data["alt_education"] = st.multiselect("Have you completed any alternative or non-traditional education?", options=other_education, accept_new_options=True, help="Include certifications, bootcamps, vocational or trade school training, etc.")

# Job info
user_data["previous_experience"] = st.radio("Do you have any previous or current work experience?", ["Yes", "No"], index=None)
if user_data["previous_experience"] == "Yes":
    st.write("Please tell us about your previous or current work experience (Optional but recommended):")
    roles = st.multiselect("Fields or roles you've worked in:", options=positions, accept_new_options=True)
    experience_list = []

    for role in roles:
        yoe = st.slider(f"Years of experience as {role}", 0, 20, 1, key=f"yoe_{role}")
        experience_list.append({"role": role, "yoe": yoe})

    user_data["experience_details"] = experience_list
    user_data["job_satisfaction"] = st.slider("On a scale of 1-10 how satisfied are you with your current job?", 0, 10, 0, 1)

# Skills
user_data["hard_skills"] = st.multiselect("Select your hard skills:", options=hard_skills, accept_new_options=True)
user_data["hard_skills_text"] = st.text_input("Other hard skills (optional):")
user_data["soft_skills"] = st.multiselect("Select your soft skills:", options=soft_skills, accept_new_options=True)
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
        return [s.strip() for s in user_data.get(field, "").split(",") if s.strip()]

    # Match hard skills
    user_hard_skills = user_data.get("hard_skills", []) + parse_text_list(user_data["hard_skills_text"])
    hard_matches = set(career["hard_skills"]).intersection(set([s.strip() for s in user_hard_skills if s.strip()]))
    score += weights["hard_skills"] * len(hard_matches)

    normalised_hard_skills = normalise_text(", ".join(user_hard_skills))
    expanded_hard_skills = expand_with_synonyms(normalised_hard_skills)

    hard_matches = set(career["hard_skills"]).intersection(expanded_hard_skills)
    score += weights["hard_skills"] * len(hard_matches)

    # Soft skills match
    user_soft_skills = user_data.get("soft_skills", []) + parse_text_list(user_data["soft_skills_text"])
    soft_matches = set(career["soft_skills"]).intersection(set([s.strip() for s in user_soft_skills if s.strip()]))
    score += weights["soft_skills"] * len(soft_matches)

    normalised_soft_skills = normalise_text(", ".join(user_soft_skills))
    expanded_soft_skills = expand_with_synonyms(normalised_soft_skills)

    soft_matches = set(career["soft_skills"]).intersection(expanded_soft_skills)
    score += weights["soft_skills"] * len(soft_matches)

    # Industry match
    field_matches = set(career["preferred_fields"]).intersection(user_data.get("interested_fields", []))
    score += weights["fields"] * len(field_matches)
    
    # College match
    try:
        college_major = user_data.get("college_major", [])
        score += weights["education"] / 2   # Flat score for college major
        major_matches = set(career["related_majors"]).intersection(set([s.strip() for s in college_major if s.strip()]))

        if major_matches:
            if "bachelor" in career.get("required_education", []):
                score += 1.5 * (weights["education"] * len(major_matches))
            else:
                score += (weights["education"] * len(major_matches))
    except AttributeError:
        print("User has not selected a college major")
    
    # Postgrad match
    try:
        postgrad_major = user_data.get("postgrad_major", [])
        score += weights["education"] / 2 # Flat score for postrad major
        postgrad_matches = set(career["related_majors"]).intersection(set([s.strip() for s in postgrad_major if s.strip]))
        
        if postgrad_matches:
            if "master+" in career["required_education"]:
                score += 1.5 * (weights["education"] * len(postgrad_matches))
            else:
                score += weights["education"] * len(postgrad_matches)
    except AttributeError:
        print("User has not selected a postgraduate major")

    # Alt education match
    try:
        alt_edu = user_data.get("alt_education", [])
        if set(alt_edu).intersection(set(career.get("alt_education", []))):
            score += weights["alt_education"]
    except AttributeError:
        print("User has not selected any alternate education")

    # Tag/Personality match
    combined_text = f"{user_data.get("likes", "")} {user_data.get("important", "")} {user_data.get("self_description", "")}".lower()
    tag_matches = [tag for tag in career["tags"] if tag.lower() in combined_text]
    score += weights["tags"] * len(tag_matches)

    # Dislikes match
    negative_text = f"{user_data.get("dislikes", "")}".lower()
    negative_matches = [tag for tag in career["tags"] if tag.lower() in negative_text]
    score -= weights["tags"] * len(negative_matches)

    # Job Satisfaction match
    job_satisfaction = user_data.get("job_satisfaction", 5)
    job_satisfaction = int(job_satisfaction)
    roles = [experience["role"] for experience in user_data.get("experience_details", [])]
    yoe = [experience["yoe"] for experience in user_data.get("experience_details", [])] # Not currently in use

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

    return score

ranked_careers = sorted(careers, key=lambda c: career_match(user_data, c), reverse=True)

# Display top matches
st.subheader("Top Career Matches:")
for i, career in enumerate(ranked_careers[:3]):
    st.markdown(f"**{i+1}. {career['title']}** — Match Score: {career_match(user_data, career)}")
