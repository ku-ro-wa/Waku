# Waku - AI-Powered Career Recommender
# Waku Video Demo - https://youtu.be/OgjspfGxZVk

Waku is an intelligent career recommendation system that analyzes user profiles (via resume upload or interactive form) and matches them against a comprehensive career database to provide personalized career path suggestions. The application combines natural language processing, semantic matching, and weighted skill analysis to deliver highly relevant recommendations.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Design Choices & Reasoning](#design-choices--reasoning)
- [Technology Stack](#technology-stack)
- [Setup Instructions](#setup-instructions)
- [Acknowledgments](#acknowledgments)

## Project Overview

Waku bridges the gap between job seekers and suitable career opportunities through intelligent analysis. Whether you upload your resume or answer a detailed questionnaire, the system extracts relevant information, understands your skills and preferences, and matches you against hundreds of career profiles using advanced NLP techniques.

This project was developed with significant assistance from AI tools. Details can be found in the #acknowledgements section of this document.

### Key Capabilities

- **Dual Input Methods**: Accept both PDF resumes and form-based input
- **Intelligent Parsing**: Extract skills, education, experience, and personality traits
- **Multi-Strategy Matching**: Combine exact matching, synonym expansion, and semantic similarity
- **Session Logging**: Automatic capture of all user data and processing steps
- **AI-Enhanced Context**: Use GPT-4 to extract certifications, achievements, and career context from resumes

## Features

- ✅ Resume PDF upload with text extraction (including OCR for scanned PDFs)
- ✅ Comprehensive form-based input for structured data collection
- ✅ Multi-algorithm matching (exact, synonym-based, semantic similarity)
- ✅ Personality and preference-aware recommendations
- ✅ Top 3 career matches with relevance scores
- ✅ AI-generated summaries of user profiles
- ✅ Complete session logging for data transparency
- ✅ Downloadable session logs (text and JSON formats)
- ✅ Automatic skill and entity extraction from text

## Project Structure

```
Waku/
├── waku/                          # Main application directory
│   ├── app.py                     # Primary Streamlit application
│   ├── requirements.txt           # Python dependencies
│   ├── README.md                  # This file
│   ├── data/                      # Career database and reference data
│   │   ├── careers.json           # Comprehensive career profiles
│   │   ├── hard_skills.json       # Hard skills reference list
│   │   ├── soft_skills.json       # Soft skills reference list
│   │   ├── majors.json            # Educational majors/fields
│   │   ├── preferred_fields.json  # Industry/field categories
│   │   ├── career_titles.json     # Job titles database
│   │   ├── alt_education.json     # Alternative education types
│   │   ├── alt_education_map.json # Mapping of alt education to careers
│   │   ├── tags.json              # Career personality tags
│   │   └── [other reference files]
│   ├── src/                       # Source code modules
│   │   ├── logger.py              # Session logging system
│   │   └── __pycache__/           # Python cache
│   └── logs/                      # Session log directory (auto-created)
│   |   ├── session_YYYYMMDD_HHMMSS.log   # Human-readable logs
│   |   └── session_YYYYMMDD_HHMMSS.json  # Structured JSON logs
|   └── utils/                     # Util files
│       ├── combine_deduplicate_json.py   # Read attributes and write to JSON files
│       └── extract_from_json.py          # Extract careers from JSON file     
|
├── LOGGING.md                     # Session logging documentation
├── SESSION_LOGGING_CHANGES.md     # Logging implementation details
└── RESUME_MATCHING_IMPROVEMENTS.md # Matching algorithm documentation
```

## File Descriptions

### Core Application Files

#### `waku/app.py` (996 lines)
The main application file built with Streamlit. Contains:

- **UI Components**: Form fields, sliders, multiselect dropdowns for data collection
- **Data Loading**: JSON data loaders for career profiles, skills, majors, etc.
- **NLP Utilities**: Text normalization, tokenization, spell-checking, synonym expansion
- **Matching Algorithms**: Multi-strategy career matching logic
- **Resume Parsing**: PDF text extraction with OCR fallback for scanned documents
- **LLM Integration**: OpenAI API calls for context extraction and summary generation
- **Logging Integration**: Session logging at all critical data points

**Key Functions**:
- `normalise_and_tokenise_text()` - Process text for analysis
- `match_user_to_targets()` - Core matching algorithm with multiple strategies
- `career_match()` - Calculate match score between user and career
- `parse_resume_to_form()` - Extract structured data from resume text
- `get_user_input_summary()` - Generate AI summaries via OpenAI API

#### `waku/src/logger.py` (290 lines)
Comprehensive session logging module that captures all activities:

- **SessionLogger Class**: Manages session creation, data logging, and file output
- **Logging Methods**: Individual logging functions for each data type
- **Dual Output**: Generates both human-readable (.log) and structured JSON logs
- **Automatic File Creation**: Creates timestamped log files in `logs/` directory
- **UTF-8 Encoding**: Handles international characters and special symbols

**Key Methods**:
- `log_user_input()` - Log individual form inputs
- `log_all_user_data()` - Log complete user data summary
- `log_pdf_upload()` - Log resume upload details
- `log_parsed_data()` - Log extracted resume information
- `log_career_matches()` - Log top career matches
- `finalize()` - Save session data to JSON and log file

### Data Files (in `waku/data/`)

All data files are JSON format, structured as key-value dictionaries or lists:

- **careers.json** - Master database with ~100+ careers, each containing:
  - Hard skills required
  - Soft skills valued
  - Related majors/fields
  - Preferred industries
  - Alternative education options
  - Personality tags
  - Experience levels (entry, mid, senior)

- **hard_skills.json** - Vocabulary of ~200+ hard skills (technical skills)
- **soft_skills.json** - Vocabulary of ~100+ soft skills (interpersonal, leadership, etc.)
- **majors.json** - ~50 educational majors and fields of study
- **preferred_fields.json** - Industry categories and fields
- **career_titles.json** - Job titles matching career profiles
- **alt_education.json** - Types of alternative education (bootcamp, certification, etc.)
- **alt_education_map.json** - Mapping which careers value specific alt education topics
- **tags.json** - Personality and characteristic tags for careers

### Documentation Files

- **LOGGING.md** - Complete guide to session logging features, formats, and usage
- **SESSION_LOGGING_CHANGES.md** - Technical summary of logging implementation
- **RESUME_MATCHING_IMPROVEMENTS.md** - Details on matching algorithms and improvements

## Design Choices & Reasoning

### 1. **Dual Input Methods (Resume + Form)**

**Choice**: Support both PDF resume uploads and detailed form inputs

**Reasoning**:
- Not everyone has an updated resume readily available
- Forms provide structured data that's easier to process consistently
- PDFs offer richer contextual information and are faster for users with recent resumes
- Streamlit's `st.file_uploader()` makes PDF integration straightforward
- Different users have different preferences; offering choice improves UX

### 2. **Multi-Strategy Matching Algorithm**

**Choice**: Combine exact matching, synonym expansion, and semantic similarity

**Reasoning**:
- **Exact Matching** (confidence: 100%) - Term-for-term matches are definitive
- **Synonym Matching** (confidence: 70-50%) - Catches variations in terminology (e.g., "ML" = "Machine Learning")
- **Semantic Matching** (confidence: 40%) - Understands meaning similarity (e.g., "leadership" ≈ "team management")
- Using all three reduces false negatives while maintaining precision
- Weighting strategies by confidence ensures high-quality matches rank higher

**Implementation**:
```python
def match_user_to_targets(user_input, target_list, weight, use_synonyms, use_semantics, ...):
    # 1. Exact matches
    # 2. Synonym expansion via WordNet
    # 3. Semantic similarity via sentence-transformers
    # Returns: (score, matched_items)
```

### 3. **Session Logging System**

**Choice**: Automatic logging of all user inputs and processing steps

**Reasoning**:
- **Data Transparency**: Users can download their complete session data
- **Audit Trail**: Track processing decisions for debugging and improvement
- **Privacy**: Data stays local in JSON format; not sent to external servers (except OpenAI for summarization)
- **Dual Format**: Text logs for human review; JSON for programmatic analysis
- **Timestamp Tracking**: Every action timestamped for sequence reconstruction

**Design Principle**: "Users own their data" - they can access, download, and review everything

### 4. **LLM Integration for Resume Context**

**Choice**: Use OpenAI GPT-4o-mini to extract certifications and achievements

**Reasoning**:
- Structured extraction (exact regex/keyword matching) misses nuanced information
- LLM can understand context and infer relationships (e.g., "led team of 5" = leadership)
- GPT-4o-mini is cost-effective while maintaining quality
- JSON response format ensures consistent parsing
- Limited to 3000 characters to control API costs

**Implementation Example**:
```
"Extract certifications, key achievements, and career context from resume"
→ Returns: { certifications: [], key_achievements: [], career_context: "" }
```

### 5. **spaCy for NLP Processing**

**Choice**: Use spaCy for tokenization, lemmatization, and entity extraction

**Reasoning**:
- **Lemmatization** - "working" = "work", "experienced" = "experience" (reduces vocabulary variance)
- **Entity Extraction** - Identify organizations, dates, Locations (GPE), languages
- **Efficient** - Optimized C backend, much faster than alternatives
- **Accurate** - Pre-trained en_core_web_sm model handles English well
- **Production-Ready** - Stable API; widely used in industry

### 6. **Weighted Skill Matching**

**Choice**: Different fields have different importance weights

```python
weights = {
    "hard_skills": 2.0,      # Most important
    "soft_skills": 1.5,
    "fields": 2.0,
    "education": 2.0,
    "alt_education": 1.0,
    "tags": 1.0              # Least important
}
```

**Reasoning**:
- Hard skills are directly job-relevant (weighted 2.0)
- Soft skills matter but are secondary (1.5)
- Educational alignment is highly important (2.0)
- Personality tags add nuance but aren't dealbreakers (1.0)
- Weights derived from domain expertise in hiring and career development

### 7. **Streamlit Framework**

**Choice**: Use Streamlit for the web interface

**Reasoning**:
- **Rapid Development** - High-level Python framework, no JavaScript needed
- **Interactive Widgets** - Built-in components (multiselect, sliders, file upload)
- **Auto-Rerun** - Updates UI when form inputs change
- **Caching** - `@st.cache_data` and `@st.cache_resource` for performance
- **Native Markdown/JSON** - Easy to display complex data
- **Perfect for ML/AI Apps** - Designed specifically for data science projects

### 8. **UTF-8 Encoding Throughout**

**Choice**: All file I/O explicitly uses `encoding='utf-8'`

**Reasoning**:
- Resumes contain international names and characters
- Default Windows encoding (charmap) fails on special characters
- UTF-8 is universal standard; works across all platforms
- Prevents `UnicodeEncodeError` and silent data loss
- Minimal performance impact; huge reliability gain

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | Interactive UI and app orchestration |
| **NLP Engine** | spaCy (en_core_web_sm) | Tokenization, lemmatization, NER |
| **Semantic Matching** | Sentence-Transformers | Embedding-based similarity matching |
| **Synonym Expansion** | NLTK WordNet | Find related terms |
| **Spell Checking** | TextBlob | Correct user input typos |
| **PDF Processing** | pdfplumber + pytesseract | Extract text from PDFs (with OCR) |
| **LLM Integration** | OpenAI GPT-4o-mini | Context extraction and summarization |
| **Logging** | Custom SessionLogger | Session tracking and data export |
| **Data Storage** | JSON files | Career database and configs |
| **Language** | Python 3.11+ | Application code |

## Setup Instructions

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Tesseract OCR (for scanned PDF support, optional)

### Installation

1. **Clone the repository**
```bash
cd c:\Users\riley\Documents\Waku
```

Github link: https://github.com/ku-ro-wa/Waku/settings

2. **Create virtual environment** (if not already created)
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
cd waku
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('omw-1.4'); nltk.download('wordnet')"
```

4. **Set up environment variables**
Create a `.env` file in the `waku/` directory:
```
OPENAI_API_KEY=your_api_key_here
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Output Files

- **Session Logs**: Saved in `logs/` directory with format `session_YYYYMMDD_HHMMSS.log` and `.json`
- **Downloads**: Users can download logs via the sidebar buttons

## Logging System

The application automatically captures all session data:

### Log Contents
- Timestamp of each action
- All user inputs (form fields, resume upload)
- Extracted data (skills, entities, achievements)
- Career match rankings
- Generated summaries
- Any errors encountered

### Log Formats
1. **Human-Readable (.log)**: Plain text with timestamps, easy to read
2. **Structured JSON (.json)**: Machine-readable; contains all data organized by type

See `LOGGING.md` for detailed logging documentation.

## Matching Algorithm Overview

The career matching system uses a sophisticated weighted scoring approach:

1. **Input Normalization**
   - Tokenize and lemmatize user input
   - Spell-correct using known vocabulary
   - Handle both list and string inputs

2. **Multi-Strategy Matching**
   ```
   For each user term:
   ├─ Exact Match (weight: 1.0) 
   ├─ Synonym Match (weight: 0.7)
   ├─ Secondary POS Synonym (weight: 0.5)
   └─ Semantic Match (weight: 0.4)
   ```

3. **Field-Weighted Scoring**
   - Hard skills: 2.0x multiplier
   - Education: 2.0x multiplier
   - Soft skills: 1.5x multiplier
   - Personality tags: 1.0x multiplier

4. **Final Ranking**
   - Careers ranked by total score
   - Top 3 presented to user with match scores

See `RESUME_MATCHING_IMPROVEMENTS.md` for algorithm details.

## Acknowledgments

### GitHub Copilot
- **Role**: Code generation, syntax assistance, function structure suggestions
- **Contribution**: Accelerated development of utility functions, form generation, and API integration
- **Usage**: Context-aware code completion throughout the project

### ChatGPT / OpenAI API
- **Role**: Natural language summarization and context extraction
- **Contribution**: GPT-4o-mini model powers resume analysis and user profile summarization
- **Implementation**: `extract_resume_context_via_llm()` function and `get_user_input_summary()`
- **Fine-tuning**: Prompt engineering for accurate, JSON-formatted extraction

### Design & Architecture
The modular design with separate `src/logger.py` and structured data files reflects best practices for:
- Maintainability: Easy to extend with new career profiles
- Testability: Separate concerns enable unit testing
- Scalability: JSON data can grow without code changes

## Possible Future Enhancements

- [ ] Skills gap analysis (what user needs to learn for target career)
- [ ] Upskilling recommendations with course suggestions
- [ ] Salary range estimation based on experience
- [ ] Career progression paths (entry → mid → senior)
- [ ] User feedback loop to improve matching accuracy
- [ ] Admin dashboard for career database management
- [ ] Multi-language support
- [ ] Integration with job boards (LinkedIn, Indeed, etc.)

## License

[Add your license here]

## Contact

For questions or suggestions, please open an issue in the repository.
Github link: https://github.com/ku-ro-wa/Waku/settings