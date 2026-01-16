import re
import os
from collections import Counter
from .resume_parser import extract_text_from_pdf, extract_text_from_docx
from .text_cleaner import clean_text
from textstat import flesch_reading_ease

# --- Advanced AI Imports ---
try:
    import spacy
    from sentence_transformers import SentenceTransformer, util
    import language_tool_python
except ImportError as e:
    print(f"AI Module Error: {e}")

# Global lazy loader
class ModelManager:
    _semantic_model = None
    _nlp = None
    _grammar_tool = None

    @classmethod
    def preload_models(cls):
        """
        Forces loading of all models at startup with logging.
        """
        print("⏳ Preloading AI Models... This may take a moment.")
        import time
        t0 = time.time()
        
        cls.get_semantic_model()
        cls.get_nlp()
        cls.get_grammar_tool()
        
        print(f"✅ All AI Models Loaded in {round(time.time() - t0, 2)} seconds.")

    @classmethod
    def get_semantic_model(cls):
        if cls._semantic_model is None:
            try:
                print("   -> Loading Semantic Model (Deep Scan: all-mpnet-base-v2)...")
                import time; s = time.time()
                cls._semantic_model = SentenceTransformer('all-mpnet-base-v2')
                print(f"      Semantic Model active ({round(time.time() - s, 2)}s)")
            except Exception as e:
                print(f"❌ Failed to load Semantic Model: {e}")
        return cls._semantic_model

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            try:
                print("   -> Loading Spacy NER Model...")
                import time; s = time.time()
                try:
                    cls._nlp = spacy.load("en_core_web_sm")
                except:
                    print("      Spacy model not found. Downloading...")
                    from spacy.cli import download
                    download("en_core_web_sm")
                    cls._nlp = spacy.load("en_core_web_sm")
                print(f"      Spacy NER active ({round(time.time() - s, 2)}s)")
            except Exception as e:
                print(f"❌ Failed to load Spacy Model: {e}")
        return cls._nlp

    @classmethod
    def get_grammar_tool(cls):
        if cls._grammar_tool is None:
            try:
                print("   -> Initializing Grammar Check (LanguageTool)...")
                import time; s = time.time()
                cls._grammar_tool = language_tool_python.LanguageTool('en-US')
                print(f"      Grammar Check active ({round(time.time() - s, 2)}s)")
            except Exception as e:
                print(f"❌ Failed to load Grammar Tool: {e}")
        return cls._grammar_tool


SECTION_PATTERNS = {
    "education": r"\beducation|academics|qualifications|scholastic|degree\b",
    "experience": r"\bexperience|work|internship|projects?|employment|history|professional\b",
    "skills": r"\bskills|technologies|tools|technical|competencies|expertise|proficiencies\b",
    "contact": r"\bemail|phone|linkedin|github|contact|address|mobile\b",
    "summary": r"\bsummary|profile|background|objective|about\b", # Added
    "certifications": r"\bcertifications?|certificates?|credentials|awards|honors\b" # Added
}

BULLET_CHARS = ("•", "-", "*", "–", "·", "➢", ">") # Expanded bullets
# --- Expanded Vocabulary (Mini-SOTA Taxonomy) ---
HARD_SKILLS = {
    "python", "java", "c++", "c#", "javascript", "typescript", "react", "angular", "vue", "node.js",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "linux", "sql", "nosql",
    "mongodb", "postgresql", "mysql", "oracle", "power bi", "tableau", "excel", "machine learning",
    "ai", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "html", "css", "api", "rest",
    "graphql", "cybersecurity", "network", "devops", "agile", "scrum", "jira"
}
SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "problem solving", "critical thinking", "adaptability",
    "time management", "creativity", "collaboration", "mentoring", "negotiation", "presentation"
}

# ------------------------
# NEW: Quantified Impact Detection (The "SOTA" Factor)
# ------------------------
def calculate_impact_score(text):
    """
    Detects 'Action Verb + Metric' patterns.
    Examples: "Increased revenue by 50%", "Reduced costs by $10k", "Managed team of 5"
    """
    if not text: return 0, []
    
    # Patterns for metrics
    patterns = [
        r"\d+%", # Percentages (e.g. 20%)
        r"\$\d+", # Money (e.g. $5000)
        r"increased|decreased|reduced|improved|generated|saved", # Action verbs
        r"\b\d+\s+(people|users|clients|customers|projects)\b" # Quantified entities
    ]
    
    impact_sentences = []
    lines = text.split('\n')
    score = 0
    
    for line in lines:
        if any(re.search(pat, line, re.I) for pat in patterns):
            # If line has BOTH a number AND an action verb, it's high impact
            if re.search(r"\d", line) and re.search(r"increased|decreased|reduced|improved|generated|saved|managed|led|created", line, re.I):
                score += 3
                impact_sentences.append(line.strip()[:100] + "...")
            elif re.search(r"\d", line):
                score += 1
    
    # Normalize score: Expecting at least 3-4 impactful impactful statements for a senior profile
    normalized_score = min(score / 15, 1.0) * 100
    return normalized_score, impact_sentences

# ------------------------
# 1. Semantic Similarity (Sentence-BERT)
# ------------------------
def calculate_semantic_score(resume_text, jd_text):
    """
    SOTA: Sliding Window Semantic Evaluation.
    Instead of truncating, we scan the entire resume in 500-char chunks and
    compare them to the JD, taking the BEST matches.
    """
    semantic_model = ModelManager.get_semantic_model()
    if not semantic_model or not jd_text: return 0.0
    
    # 1. Chunking 
    # Resume is often long. We split it into 500-char overlapping segments.
    chunk_size = 500
    overlap = 100
    chunks = []
    
    for i in range(0, len(resume_text), chunk_size - overlap):
        chunks.append(resume_text[i:i + chunk_size])
        
    if not chunks: chunks = [resume_text]
    
    # 2. Encode JD (Target)
    # JD is also potentially long, but let's take the first 1500 chars (usually sufficient)
    jd_embedding = semantic_model.encode(jd_text[:1500])
    
    # 3. Encode All Resume Chunks
    chunk_embeddings = semantic_model.encode(chunks)
    
    # 4. Compute Similarities
    # We get a list of scores.
    cosine_scores = util.cos_sim(jd_embedding, chunk_embeddings)[0]
    
    # 5. Top-K Aggregation
    # We take the average of the Top 3 best matching segments. 
    # This prevents one bad section from ruining the score.
    # If fewer than 3 chunks, take average of all.
    top_k_scores = cosine_scores.topk(min(3, len(chunks))).values
    final_score = top_k_scores.mean().item()
    
    return final_score * 100 

# ------------------------
# 2. Entity Extraction (NER)
# ------------------------
def extract_entities(text):
    nlp = ModelManager.get_nlp()
    if not nlp: return {}
    # Optimization: Limit NER to first 2500 chars to speed up processing
    doc = nlp(text[:2500])
    entities = {}
    
    # Extract Organizations (Companies)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    # Extract Dates (Experience duration candidates)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    
    entities['org_count'] = len(set(orgs))
    entities['date_count'] = len(dates)
    entities['top_orgs'] = list(set(orgs))[:3]
    return entities

# ------------------------
# 3. Grammar Check
# ------------------------
def check_grammar(text):
    grammar_tool = ModelManager.get_grammar_tool()
    if not grammar_tool: return [], 0
    # Optimization: Reduce window to 600 chars (Summary/Intro) for blazing speed
    matches = grammar_tool.check(text[:600])
    errors = [match.message for match in matches[:3]] # Top 3 errors
    return errors, len(matches)

# ------------------------
# Helper Utils
# ------------------------
def detect_sections(text):
    return {sec: bool(re.search(pat, text, re.I)) for sec, pat in SECTION_PATTERNS.items()}

def extract_keywords_from_jd(jd_text):
    """
    SOTA: Use Spacy NLP to extract only relevant Skills/Tech (PROPN, NOUN).
    Filters out common stopwords and generic words.
    """
    nlp = ModelManager.get_nlp()
    if not nlp:
        # Fallback to basic extraction if NLP fails
        cleaned = clean_text(jd_text)
        return list(set(w for w in cleaned.split() if len(w)>2))
    
    doc = nlp(jd_text)
    keywords = set()
    
    # NLP Strategy: Keep Proper Nouns (Python, AWS) and specific Nouns (Database, CI/CD)
    # Filter out stopwords (and, the, a) and generic nouns (candidate, work, experience)
    GENERIC_TERMS = {"experience", "candidate", "work", "year", "team", "role", "skill", "ability", "knowledge", "degree", "job", "company", "opportunity", "requirement", "qualification", "proficiency", "understanding"}
    
    for token in doc:
        if token.is_stop or token.is_punct or len(token.text) < 2:
            continue
            
        # Strategy A: Proper Nouns (Most tech stacks are PROPN)
        if token.pos_ == "PROPN":
            keywords.add(token.text)
            
        # Strategy B: Noun Chunks or specific nouns (often soft skills or concepts)
        elif token.pos_ == "NOUN" and token.lemma_.lower() not in GENERIC_TERMS:
            # Only add if it looks "technical" or specific (heuristic)
            if len(token.text) > 3: 
                keywords.add(token.text)
                
    return list(keywords)

def extract_keywords(text, keywords):
    text_lower = text.lower()
    # Improvement: Whole word matching prevents "Java" matching inside "Javascript" matches (imperfect but better)
    # For now, simplistic check is faster and usually sufficient for broad matches
    present = [kw for kw in keywords if kw.lower() in text_lower]
    missing = [kw for kw in keywords if kw not in present]
    return present, missing

def bullet_ratio(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: return 0
    bullets = sum(1 for l in lines if l[0] in BULLET_CHARS)
    return bullets / len(lines)

def generate_priorities(scores, metrics):
    """
    Generates the 'Top 3 Priorities' list dynamically based on analysis results.
    """
    priorities = []
    
    # 1. Critical: Job Description Mismatch
    if scores['semantic'] < 50 and metrics['has_jd']:
        priorities.append({
            'icon': 'fa-triangle-exclamation',
            'color': 'text-red-400',
            'text': 'Resume content does not match the Job Description. Rewrite summaries and experience to align with the JD.'
        })
        
    # 2. High: Missing Keywords
    if metrics['missing_keywords']:
        count = len(metrics['missing_keywords'])
        # Show top 3 missing words in the text to be helpful
        top_missing = ", ".join(metrics['missing_keywords'][:3])
        priorities.append({
            'icon': 'fa-key',
            'color': 'text-red-400',
            'text': f'Add missing skills: {top_missing}' + (f' and {count-3} more...' if count > 3 else '.')
        })
        
    # 3. Medium: Grammar
    if metrics['grammar_issues']:
        count = len(metrics['grammar_issues'])
        priorities.append({
            'icon': 'fa-spell-check',
            'color': 'text-amber-400',
            'text': f'Fix {count} grammar/style errors found in the text.'
        })
        
    # 4. Low: Structure/Impact
    if metrics['impact_score'] < 40:
        priorities.append({
            'icon': 'fa-chart-line',
            'color': 'text-blue-400',
            'text': 'Quantify your experience. Add numbers (e.g., "Improved by 20%") to your bullet points.'
        })
        
    # 5. Bonus: Missing Sections
    missing_secs = [k for k,v in metrics['sections'].items() if not v]
    if missing_secs:
         priorities.append({
            'icon': 'fa-pen',
            'color': 'text-gray-400',
            'text': f'Add missing section: {missing_secs[0].capitalize()}'
        })

    # Fallback if perfect
    if not priorities:
        priorities.append({
            'icon': 'fa-rocket',
            'color': 'text-green-400',
            'text': 'Resume looks great! Consider applying now.'
        })
        
    return priorities

# --- RLHF Calibration ---
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'calibration.json')

def load_calibration():
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"strictness_bias": 1.0, "semantic_weight": 40, "learning_rate": 0.05}

def update_calibration(feedback_type):
    """
    Self-Improving function: Adjusts weights based on user feedback.
    feedback_type: 'too_low', 'accurate', 'too_high'
    """
    data = load_calibration()
    lr = data.get("learning_rate", 0.05)
    
    if feedback_type == 'too_low':
        # User says we are too harsh. Relax the bias.
        data['strictness_bias'] = max(0.5, data['strictness_bias'] - lr)
        # Boost semantic score importance (giving more benefit of doubt)
        data['semantic_weight'] = min(60, data.get('semantic_weight', 40) + 1)
        
    elif feedback_type == 'too_high':
        # User says we are too generous. Tighten the bias.
        data['strictness_bias'] = min(1.5, data['strictness_bias'] + lr)
        
    elif feedback_type == 'accurate':
        # Reinforcement: slightly decrease learning rate (converging)
        data['learning_rate'] = max(0.01, lr * 0.99)

    data['total_feedback_count'] = data.get('total_feedback_count', 0) + 1
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    return data

# ------------------------
# 4. Weakness Detection (Deep Dive)
# ------------------------
def extract_weak_bullets(text):
    """
    Identifies 'weak' lines that need rewriting.
    Criteria:
    - Starts with weak verbs ("Worked on", "Responsible for", "Assisted")
    - Too short (< 6 words)
    - Vague (no numbers, no strong nouns)
    """
    weak_starters = ["worked", "responsible", "assisted", "helped", "handled", "participated", "involved", "various"]
    
    weak_bullets = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10: continue # Skip headers/empty
        
        # Check if it looks like a bullet point or sentence
        is_bullet = line[0] in BULLET_CHARS or line[0] == "-" or (len(line) > 20 and line[0].isupper())
        
        if is_bullet:
            words = line.split()
            # 1. Too Short
            if len(words) < 6:
                weak_bullets.append({"text": line, "reason": "Too short. Expand with details."})
                continue
                
            # 2. Weak Starter
            first_word = words[0].lower()
            if first_word in weak_starters or (len(words)>1 and f"{first_word} {words[1].lower()}" in ["worked on", "responsible for"]):
                weak_bullets.append({"text": line, "reason": "Passive/Weak verb. Use 'Led', 'Built', 'Optimized'."})
                continue
                
            # 3. Vague (Middle of text check)
            if "responsibility" in line.lower() or "various tasks" in line.lower():
                 weak_bullets.append({"text": line, "reason": "Vague. Specify what tasks you did."})
    
    # Return unique top 5
    # Remove duplicates preserving order
    unique = []
    seen = set()
    for w in weak_bullets:
        if w['text'] not in seen:
            unique.append(w)
            seen.add(w['text'])
            
    return unique[:5]


# ------------------------
# 5. Helper Scoring Functions (Strict Mode)
# ------------------------
def calculate_formatting_score(text):
    """
    Checks formatting (10% weight).
    Criteria: Bullet points usage, readable sentence length.
    """
    issues = []
    score = 100
    
    # 1. Bullet Points
    b_ratio = bullet_ratio(text)
    if b_ratio < 0.1:
        score -= 50
        issues.append("Too few bullet points. Use bullets for lists.")
    elif b_ratio > 0.6:
        score -= 20
        issues.append("Too many bullet points. Balance with paragraphs.")
        
    # 2. Sentence Length (Average)
    sentences = [s for s in text.split('.') if len(s.split()) > 3]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len > 25:
            score -= 30
            issues.append("Sentences are too long (>25 words). Shorten them.")
        elif avg_len < 8:
            score -= 10
            issues.append("Sentences look too choppy/short.")
            
    return max(0, score), issues

def calculate_contact_score(text, entities):
    """
    Checks Contact Info (5% weight).
    Criteria: Email, Phone, Links.
    """
    score = 0
    issues = []
    
    # Simple RegEx checks
    import re
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    link_pattern = r'linkedin\.com|github\.com|portfolio|website'
    
    if re.search(email_pattern, text):
        score += 40
    else:
        issues.append("Missing Email Address.")
        
    if re.search(phone_pattern, text):
        score += 30
    else:
        issues.append("Missing Phone Number.")
        
    if re.search(link_pattern, text, re.IGNORECASE):
        score += 30
    else:
        issues.append("Missing LinkedIn or Portfolio Link.")
        
    return min(100, score), issues

def calculate_education_score(sections):
    """
    Checks Education (10% weight).
    """
    score = 0
    issues = []
    if sections.get('education'):
        score = 100
    else:
        issues.append("Missing 'Education' section.")
    return score, issues


# ------------------------
# Main Computation (Refactored for Strict Rules)
# ------------------------
def compute_score(text, jd_text=""):
    # Load dynamic RLHF weights (We override them with strict user rules, but keep bias)
    brain = load_calibration()
    bias = brain.get('strictness_bias', 1.0)
    
    sections = detect_sections(text)
    feedback = []
    deep_dive = {} # Dictionary to hold specific feedback for each category
    
    # --- 1. Keyword Match (45%) ---
    # Logic: (Semantic Score + Exact Keyword Match) / 2
    semantic_model = ModelManager.get_semantic_model()
    
    if jd_text:
        # A. Semantic
        if semantic_model:
            sim_score = calculate_semantic_score(text, jd_text)
        else:
            sim_score = 0
        
        # B. Exact Match
        if "," in jd_text:
             jd_keywords = [k.strip() for k in jd_text.split(",") if k.strip()]
        else:
             jd_keywords = extract_keywords_from_jd(jd_text)
        
        present_kw, missing_kw = extract_keywords(text, jd_keywords)
        match_ratio = (len(present_kw) / max(1, len(jd_keywords))) * 100
        
        keyword_score = (sim_score + match_ratio) / 2
        keyword_score = min(keyword_score * (1/bias), 100)
    else:
        # General Mode
        # Restore basic extraction for general mode
        found_hard, _ = extract_keywords(text, HARD_SKILLS)
        found_soft, _ = extract_keywords(text, SOFT_SKILLS)
        present_kw = list(set(found_hard + found_soft))
        
        # Simple scoring for general mode: if > 15 skills found -> 100%
        keyword_score = min((len(present_kw) / 15) * 100, 100)
        
        missing_kw = []
        feedback.append("General Analysis: Found " + str(len(present_kw)) + " skills.")
    
    deep_dive['keywords'] = []
    if keyword_score < 70 and jd_text:
         deep_dive['keywords'].append(f"Only matched {len(present_kw)}/{len(jd_keywords)} important keywords.")

    # --- 2. Skills Section (15%) ---
    # Logic: Exists (50) + Quantity > 5 (50)
    skills_score = 0
    start_score_skills = 0
    skills_issues = []
    
    if sections.get('skills'):
        skills_score += 50
        # Fix: sections['skills'] is a boolean, so we can't split it. 
        # Instead, we check if we found enough keywords in the whole text as a proxy.
        # It's an approximation but robust.
        all_skills_found, _ = extract_keywords(text, HARD_SKILLS.union(SOFT_SKILLS))
        
        if len(all_skills_found) > 5:
            skills_score += 50
        else:
            skills_issues.append("List more than 5 specific skills.")
    else:
        skills_issues.append("Missing 'Skills' section.")
        
    deep_dive['skills'] = skills_issues

    # --- 3. Experience/Projects (15%) ---
    # Logic: Exists (50) + Impact Score (50)
    exp_score = 0
    exp_issues = []
    
    if sections.get('experience') or sections.get('projects'):
        exp_score += 50
    else:
        exp_issues.append("Missing 'Experience' or 'Projects' section.")
        
    impact_raw, impacts = calculate_impact_score(text)
    # Impact score is already 0-100. We want it to contribute 50pts max.
    exp_score += min(impact_raw, 100) * 0.5
    
    if impact_raw < 40:
        exp_issues.append("Low Impact: Use more numbers/metrics in your bullets.")
        
    deep_dive['experience'] = exp_issues

    # --- 4. Formatting (10%) --- 
    fmt_score, fmt_issues = calculate_formatting_score(text)
    deep_dive['formatting'] = fmt_issues

    # --- 5. Education (10%) ---
    edu_score, edu_issues = calculate_education_score(sections)
    deep_dive['education'] = edu_issues

    # --- 6. Contact Info (5%) ---
    contact_score, contact_issues = calculate_contact_score(text, extract_entities(text))
    deep_dive['contact'] = contact_issues


    # --- FINAL CALCULATION ---
    # Weights: 45, 15, 15, 10, 10, 5
    w_kw = 0.45
    w_sk = 0.15
    w_ex = 0.15
    w_fm = 0.10
    w_ed = 0.10
    w_co = 0.05
    
    total = (
        (keyword_score * w_kw) +
        (skills_score * w_sk) +
        (exp_score * w_ex) +
        (fmt_score * w_fm) +
        (edu_score * w_ed) +
        (contact_score * w_co)
    )
    
    total = min(round(total), 100) # Cap at 100
    
    # Store breakdown for frontend (Raw scores 0-100)
    breakdown = {
        "keywords": int(keyword_score),
        "skills": int(skills_score),
        "experience": int(exp_score),
        "formatting": int(fmt_score),
        "education": int(edu_score),
        "contact": int(contact_score)
    }

    # Helper: Generate Priorities based on LOWEST scores
    # We create a simple priority list
    priorities = []
    
    # Check lowest categories first
    sorted_cats = sorted(breakdown.items(), key=lambda x: x[1])
    
    for cat, score in sorted_cats:
        if score < 100:
            # Add specific advice from Deep Dive
            issues_list = deep_dive.get(cat, [])
            if issues_list:
                priorities.append({
                    'icon': 'fa-triangle-exclamation',
                    'color': 'text-red-400', 
                    'text': f"{cat.capitalize()}: {issues_list[0]}"
                })
    
    # Add grammar fallback
    grammar_errors, _ = check_grammar(text)
    if len(grammar_errors) > 3:
        priorities.append({
            'icon': 'fa-spell-check',
            'color': 'text-amber-400',
            'text': f"Fix {len(grammar_errors)} grammar errors."
        })

    # Get weak bullets for legacy support/extra display
    weak_bullets = extract_weak_bullets(text)

    return {
        "total_score": int(total),
        "breakdown": breakdown,
        "sections": sections,
        "present_keywords": present_kw,
        "missing_keywords": missing_kw,
        "feedback": feedback,
        "entities": {}, # Not strictly used in new logic but good to keep
        "grammar_issues": grammar_errors,
        "calibration": brain, 
        "impacts": impacts, 
        "weak_bullets": weak_bullets,
        "priorities": priorities[:4], # Top 4
        "deep_dive": deep_dive # NEW: Full detailed feedback
    }

