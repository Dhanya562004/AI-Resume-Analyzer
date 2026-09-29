import streamlit as st
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Resume Analyzer & ATS Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for visual excellence
st.markdown("""
<style>
    /* Main container background and text styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366F1;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
    }
    .tag-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .tag-matched {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .tag-missing {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .tag-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .section-card {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #6366F1;
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sample Data for 1-click Demo Testing
SAMPLE_RESUME = """ALEX JOHNSON
Software Engineer | Full Stack Developer
Email: alex.johnson@email.com | LinkedIn: linkedin.com/in/alexjohnson | GitHub: github.com/alexjohnson

PROFESSIONAL SUMMARY
Results-driven Software Engineer with 3+ years of experience building scalable web applications, REST APIs, and microservices using Python, JavaScript, React, and SQL. Skilled in cloud deployment (AWS, Docker), Agile methodology, and CI/CD pipelines.

TECHNICAL SKILLS
- Programming Languages: Python, JavaScript, TypeScript, SQL, HTML5, CSS3
- Frameworks & Libraries: React, Node.js, Express, FastAPI, Flask, Django, Streamlit
- Databases: PostgreSQL, MongoDB, Redis, MySQL
- DevOps & Tools: AWS (S3, EC2), Docker, Git, GitHub Actions, Linux, JIRA, Postman

WORK EXPERIENCE
Software Engineer | TechCorp Solutions (2022 - Present)
- Developed and maintained responsive web applications serving 50,000+ monthly active users using React and Python FastAPI.
- Designed and optimized PostgreSQL database queries, reducing API response times by 35%.
- Implemented CI/CD automation pipelines with GitHub Actions, accelerating release cycles by 40%.
- Collaborated with cross-functional teams in an Agile/Scrum environment to deliver sprint goals on schedule.

Junior Web Developer | InnovateX Labs (2021 - 2022)
- Built interactive user interfaces using React and Redux, improving user engagement metrics.
- Created microservices and RESTful API endpoints with Node.js and Express.
- Conducted unit testing and code reviews to ensure high software quality standards.

EDUCATION
Bachelor of Science in Computer Science | University of Technology (2017 - 2021)

PROJECTS
- AI Resume Analyzer: Streamlit app leveraging NLP algorithms to analyze resume-JD match score.
- E-Commerce Platform: Full-stack web app with React, Node.js, and MongoDB featuring payment integration."""

SAMPLE_JD = """We are looking for a Senior Software Engineer / Full Stack Developer to join our core engineering team.

Key Responsibilities:
- Build and scale robust backend RESTful APIs using Python, FastAPI, or Node.js.
- Develop interactive frontend interfaces using React, TypeScript, and modern CSS frameworks.
- Architect scalable database schemas with PostgreSQL and MongoDB.
- Containerize applications using Docker and deploy services on AWS cloud infrastructure.
- Implement CI/CD pipelines and automated testing suites to maintain high code reliability.
- Participate in Agile ceremonies, code reviews, and technical design discussions.

Required Qualifications & Skills:
- 3+ years of professional software engineering experience.
- Strong proficiency in Python, JavaScript, TypeScript, React, and SQL.
- Hands-on experience with cloud infrastructure (AWS) and containerization (Docker, Kubernetes).
- Familiarity with CI/CD tools, Git, Linux administration, and system design.
- Excellent problem-solving, communication, and collaboration skills.
- Bachelor's degree in Computer Science or equivalent field."""

# STOPWORDS for NLP Processing
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "experience", "years", "work", "job", "description", "ability", "strong", "knowledge", "required", "preferred",
    "responsibilities", "qualifications", "looking", "candidate", "role"
}

# Domain Skill Taxonomy for Tagging
KNOWN_SKILLS = {
    # Tech / Programming
    "python", "javascript", "typescript", "java", "c++", "c#", "ruby", "php", "go", "golang", "rust", "scala", "kotlin",
    "swift", "html", "html5", "css", "css3", "sql", "nosql", "bash", "powershell", "shell", "r", "matlab",
    # Frameworks & Web
    "react", "angular", "vue", "next.js", "node", "node.js", "express", "fastapi", "flask",
    "django", "spring", "springboot", "dotnet", ".net", "streamlit", "bootstrap", "tailwindcss", "redux",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "jenkins", "git", "github", "gitlab", "ci/cd", "terraform",
    "ansible", "cloud", "linux", "unix", "nginx", "apache",
    # Databases
    "postgresql", "postgres", "mongodb", "mysql", "sqlite", "redis", "elasticsearch", "dynamodb", "oracle",
    # Data & AI
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "opencv", "nlp", "cv", "machine learning",
    "deep learning", "artificial intelligence", "data science", "data analysis", "power bi", "tableau", "spark", "hadoop",
    # Methodology & Tools
    "agile", "scrum", "kanban", "jira", "rest", "restful", "api", "graphql", "microservices", "unit testing", "pytest",
    "leadership", "communication", "problem solving", "collaboration", "teamwork", "management", "architecture"
}

# PDF Text Extraction Function with Multi-library Fallbacks
def extract_text_from_pdf(uploaded_file):
    text = ""
    if uploaded_file is None:
        return text

    # Attempt 1: pypdf
    try:
        import pypdf
        uploaded_file.seek(0)
        reader = pypdf.PdfReader(uploaded_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            return text.strip()
    except Exception:
        pass

    # Attempt 2: PyPDF2
    try:
        import PyPDF2
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            return text.strip()
    except Exception:
        pass

    # Attempt 3: pdfplumber
    try:
        import pdfplumber
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text.strip()
    except Exception:
        pass

    # Fallback to UTF-8 decoding
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
        return content.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""

def extract_tokens(text):
    clean = re.sub(r'[^a-zA-Z0-9\+\#\.]', ' ', text.lower())
    tokens = [t.strip('.') for t in clean.split() if len(t) > 1 and t not in STOPWORDS]
    return tokens

# Core AI Analysis Engine
def analyze_resume_jd(resume_text, jd_text):
    if not resume_text.strip() or not jd_text.strip():
        return None

    resume_tokens = extract_tokens(resume_text)
    jd_tokens = extract_tokens(jd_text)

    resume_set = set(resume_tokens)
    jd_set = set(jd_tokens)

    # 1. Cosine Similarity via TF-IDF
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
        cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        tfidf_score = float(cos_sim * 100)
    except Exception:
        tfidf_score = 0.0

    # 2. Skill matching
    jd_skills = {word for word in jd_set if word in KNOWN_SKILLS or any(word in s for s in KNOWN_SKILLS)}
    resume_skills = {word for word in resume_set if word in KNOWN_SKILLS or any(word in s for s in KNOWN_SKILLS)}

    if not jd_skills:
        jd_skills = jd_set

    matched_skills = jd_skills.intersection(resume_set)
    missing_skills = jd_skills - matched_skills

    skill_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0.0

    # 3. Overall Keyword Match
    matched_all_keywords = jd_set.intersection(resume_set)
    missing_all_keywords = jd_set - resume_set
    keyword_score = (len(matched_all_keywords) / len(jd_set) * 100) if jd_set else 0.0

    # Combined Score Formula
    overall_score = round((tfidf_score * 0.4) + (skill_score * 0.4) + (keyword_score * 0.2), 1)
    overall_score = min(100.0, max(0.0, overall_score))

    # Match Verdict Classification
    if overall_score >= 80:
        verdict = "🔥 Excellent Match"
        verdict_color = "#10B981"
        verdict_desc = "Your resume is highly aligned with the job description. Excellent keyword and technical coverage!"
    elif overall_score >= 60:
        verdict = "✅ Good Match"
        verdict_color = "#3B82F6"
        verdict_desc = "Solid match with good core alignment. Addressing missing skills can significantly boost your ATS score."
    elif overall_score >= 40:
        verdict = "⚠️ Moderate Match"
        verdict_color = "#F59E0B"
        verdict_desc = "Partial alignment found. Key domain skills and tools are missing from your resume."
    else:
        verdict = "❌ Needs Improvement"
        verdict_color = "#EF4444"
        verdict_desc = "Low similarity detected. Tailor your resume specifically to include keywords from the job description."

    # ATS Checks & Audit
    word_count = len(resume_text.split())
    has_summary = any(k in resume_text.lower() for k in ["summary", "profile", "objective", "about me"])
    has_experience = any(k in resume_text.lower() for k in ["experience", "employment", "work history", "work experience"])
    has_education = any(k in resume_text.lower() for k in ["education", "academic", "university", "college", "degree"])
    has_skills = any(k in resume_text.lower() for k in ["skills", "technologies", "competencies", "expertise"])
    has_projects = any(k in resume_text.lower() for k in ["projects", "portfolio"])
    bullet_count = len(re.findall(r'[\n\r]\s*[\-\•\*\d+\.]\s+', resume_text))

    ats_score = 100
    ats_issues = []
    if word_count < 200:
        ats_score -= 20
        ats_issues.append("Resume word count is too short (< 200 words). Add more details regarding your projects and experience.")
    elif word_count > 1200:
        ats_score -= 10
        ats_issues.append("Resume is quite lengthy (> 1200 words). Aim for concise 1-2 page formatting.")

    if not has_summary:
        ats_score -= 10
        ats_issues.append("Missing explicit 'Summary' or 'Professional Profile' section heading.")
    if not has_experience:
        ats_score -= 20
        ats_issues.append("Missing explicit 'Work Experience' section heading.")
    if not has_skills:
        ats_score -= 15
        ats_issues.append("Missing explicit 'Skills' section heading.")
    if not has_education:
        ats_score -= 10
        ats_issues.append("Missing explicit 'Education' section heading.")
    if bullet_count < 3:
        ats_score -= 15
        ats_issues.append("Low bullet-point density. Format your achievements with bullet points for higher ATS readability.")

    ats_score = max(0, ats_score)

    return {
        "overall_score": overall_score,
        "tfidf_score": round(tfidf_score, 1),
        "skill_score": round(skill_score, 1),
        "keyword_score": round(keyword_score, 1),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_desc": verdict_desc,
        "matched_skills": sorted(list(matched_skills)),
        "missing_skills": sorted(list(missing_skills)),
        "matched_keywords_count": len(matched_all_keywords),
        "missing_keywords_count": len(missing_all_keywords),
        "top_missing_keywords": sorted(list(missing_all_keywords))[:15],
        "ats_score": ats_score,
        "ats_issues": ats_issues,
        "word_count": word_count,
        "bullet_count": bullet_count,
        "sections": {
            "Summary": has_summary,
            "Experience": has_experience,
            "Education": has_education,
            "Skills": has_skills,
            "Projects": has_projects
        }
    }

# Plotly Charts
def create_gauge_chart(score, verdict_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match Compatibility Score", 'font': {'size': 18, 'color': '#FFFFFF'}},
        number={'suffix': "%", 'font': {'size': 42, 'color': verdict_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#666666"},
            'bar': {'color': verdict_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333333",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FFFFFF"}
    )
    return fig

def create_skills_chart(matched_count, missing_count):
    df = pd.DataFrame({
        'Category': ['Matched Skills', 'Missing Skills'],
        'Count': [matched_count, missing_count],
        'Color': ['#10B981', '#EF4444']
    })
    fig = px.bar(
        df,
        x='Category',
        y='Count',
        color='Category',
        color_discrete_map={'Matched Skills': '#10B981', 'Missing Skills': '#EF4444'},
        text='Count',
        title='Skill Match Breakdown'
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(color="#FFFFFF")
    )
    fig.update_traces(textposition='outside')
    return fig

# Streamlit Session State Initialization
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""

# Sidebar Controls & Information
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/resume.png", width=70)
    st.title("AI Resume Analyzer")
    st.caption("GenAI-Powered ATS Matcher & Optimizer")
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Demo Loaders")
    if st.button("📋 Load Sample Software Engineer Data", use_container_width=True):
        st.session_state["resume_text"] = SAMPLE_RESUME
        st.session_state["jd_text"] = SAMPLE_JD
        st.rerun()

    if st.button("🗑️ Clear Inputs", use_container_width=True):
        st.session_state["resume_text"] = ""
        st.session_state["jd_text"] = ""
        if "analysis" in st.session_state:
            del st.session_state["analysis"]
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Features")
    st.markdown("""
    - 📄 **PDF & Text Parsing**
    - 📊 **Match Compatibility Score**
    - 🏷️ **Matched vs Missing Skills**
    - 🤖 **ATS Formatting Audit**
    - 💡 **Tailored Improvement Tips**
    """)
    
    st.markdown("---")
    st.markdown("🌐 [Live Web App](https://ai-resume-analyzer-npmgmltfplr6r4ayqbfzvg.streamlit.app/)")
    st.markdown("🐙 [GitHub Repo](https://github.com/Dhanya562004/AI-Resume-Analyzer)")

# Main Header
st.markdown('<div class="main-header">📄 AI Resume Analyzer & ATS Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your resume PDF or paste text to evaluate ATS compatibility and skill matching against any Job Description.</div>', unsafe_allow_html=True)

# Main Input Section: 2 Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Resume Input")
    input_method = st.radio("Choose Input Method:", ["📄 Upload PDF", "✏️ Paste Raw Text"], horizontal=True)

    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf", "txt"])
        if uploaded_file is not None:
            extracted = extract_text_from_pdf(uploaded_file)
            if extracted:
                st.session_state["resume_text"] = extracted
                st.success(f"Successfully parsed '{uploaded_file.name}' ({len(extracted.split())} words)")
            else:
                st.warning("⚠️ Could not extract text from PDF (it might be a scanned image). Please switch to 'Paste Raw Text' and enter your resume manually.")

    # Always show editable text area pre-filled with state
    resume_input = st.text_area(
        "Resume Content:",
        value=st.session_state.get("resume_text", ""),
        height=320,
        placeholder="Paste your resume content here or upload a PDF above...",
        key="resume_textarea"
    )
    st.session_state["resume_text"] = resume_input

with col2:
    st.subheader("2. Job Description")
    st.markdown("&nbsp;", unsafe_allow_html=True) # align layout height
    jd_input = st.text_area(
        "Job Description Content:",
        value=st.session_state.get("jd_text", ""),
        height=370,
        placeholder="Paste the target job description requirements and responsibilities here...",
        key="jd_textarea"
    )
    st.session_state["jd_text"] = jd_input

# Analyze Action Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Analyze Resume & Match Job", type="primary", use_container_width=True):
    if not st.session_state["resume_text"].strip() or not st.session_state["jd_text"].strip():
        st.warning("⚠️ Please provide both a Resume and a Job Description before running analysis.")
    else:
        with st.spinner("Analyzing resume content, extracting skills, and calculating match compatibility..."):
            st.session_state["analysis"] = analyze_resume_jd(
                st.session_state["resume_text"],
                st.session_state["jd_text"]
            )

# Display Analysis Output
if "analysis" in st.session_state and st.session_state["analysis"] is not None:
    res = st.session_state["analysis"]
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Dashboard")

    # High-level Metrics Row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Match</div>
            <div class="metric-value" style="color:{res['verdict_color']}">{res['overall_score']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ATS Audit Score</div>
            <div class="metric-value" style="color: {'#10B981' if res['ats_score'] >= 80 else '#F59E0B'}">{res['ats_score']}/100</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Skill Match Rate</div>
            <div class="metric-value">{res['skill_score']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Matched Keywords</div>
            <div class="metric-value" style="color:#10B981">{res['matched_keywords_count']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Overview & Verdict",
        "🏷️ Skill & Gap Analysis",
        "🤖 ATS Compliance Audit",
        "💡 Actionable AI Recommendations"
    ])

    with tab1:
        tcol1, tcol2 = st.columns([1, 1])
        with tcol1:
            st.plotly_chart(create_gauge_chart(res['overall_score'], res['verdict_color']), use_container_width=True)
        with tcol2:
            st.markdown(f"### Verdict: {res['verdict']}")
            st.info(res['verdict_desc'])
            st.markdown(f"""
            - **TF-IDF Semantic Similarity Score**: `{res['tfidf_score']}%`
            - **Technical Skill Match Score**: `{res['skill_score']}%`
            - **Overall Keyword Overlap**: `{res['keyword_score']}%`
            - **Total Word Count**: `{res['word_count']} words`
            """)

    with tab2:
        scol1, scol2 = st.columns([1, 1])
        with scol1:
            st.plotly_chart(create_skills_chart(len(res['matched_skills']), len(res['missing_skills'])), use_container_width=True)
        
        with scol2:
            st.markdown("### 🚀 Matched Skills")
            if res['matched_skills']:
                matched_html = "".join([f'<span class="tag-pill tag-matched">✓ {skill.title()}</span>' for skill in res['matched_skills']])
                st.markdown(matched_html, unsafe_allow_html=True)
            else:
                st.write("No direct skill matches detected.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### ❌ Missing Critical Skills")
            if res['missing_skills']:
                missing_html = "".join([f'<span class="tag-pill tag-missing">✗ {skill.title()}</span>' for skill in res['missing_skills']])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("Great job! No key skills missing from the job description.")

        if res['top_missing_keywords']:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔍 Top Missing Keywords in Job Description")
            missing_kw_html = "".join([f'<span class="tag-pill tag-info">{kw}</span>' for kw in res['top_missing_keywords']])
            st.markdown(missing_kw_html, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 📋 Resume Structure & ATS Readiness Check")
        st.write(f"**Overall ATS Score: {res['ats_score']} / 100**")
        
        seccol1, seccol2 = st.columns(2)
        with seccol1:
            st.markdown("#### Essential Sections Detected:")
            for sec_name, present in res['sections'].items():
                if present:
                    st.markdown(f"✅ **{sec_name} Section**: Detected")
                else:
                    st.markdown(f"❌ **{sec_name} Section**: Missing heading")

        with seccol2:
            st.markdown("#### Formatting Metrics:")
            st.markdown(f"• **Word Count**: {res['word_count']} words (Optimal: 300 - 800)")
            st.markdown(f"• **Bullet Points**: {res['bullet_count']} bullet items found")

        if res['ats_issues']:
            st.markdown("#### ⚠️ Potential ATS Formatting Concerns:")
            for issue in res['ats_issues']:
                st.warning(f"• {issue}")
        else:
            st.success("🎉 No major ATS formatting issues detected!")

    with tab4:
        st.markdown("### 💡 Recommended Action Steps to Improve Your Resume")
        
        st.markdown("""
        <div class="section-card">
            <h4>1. Integrate Missing Keywords</h4>
            <p>Naturally include missing skills and terms identified in the Gap Analysis tab into your Work Experience bullet points and Skills section.</p>
        </div>
        <div class="section-card">
            <h4>2. Quantify Your Achievements</h4>
            <p>Enhance bullet points by adding measurable outcomes. For example, replace <i>'Improved application performance'</i> with <i>'Optimized database queries, reducing API response times by 35%'</i>.</p>
        </div>
        <div class="section-card">
            <h4>3. Use Strong Action Verbs</h4>
            <p>Begin every experience bullet point with impactful action verbs like: <b>Architected, Spearheaded, Implemented, Automated, Engineered, Optimized, Streamlined</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        # Download Report Feature
        report_text = f"""==================================================
AI RESUME ANALYZER & ATS EVALUATION REPORT
==================================================
Overall Match Score: {res['overall_score']}%
Verdict: {res['verdict']}
ATS Readiness Score: {res['ats_score']}/100

SUMMARY BREAKDOWN:
--------------------------------------------------
- TF-IDF Cosine Similarity: {res['tfidf_score']}%
- Skill Match Rate: {res['skill_score']}%
- Keyword Overlap Rate: {res['keyword_score']}%
- Resume Word Count: {res['word_count']} words

MATCHED SKILLS:
--------------------------------------------------
{', '.join(res['matched_skills']) if res['matched_skills'] else 'None'}

MISSING SKILLS:
--------------------------------------------------
{', '.join(res['missing_skills']) if res['missing_skills'] else 'None'}

ATS ISSUES & AUDIT:
--------------------------------------------------
{chr(10).join(['- ' + issue for issue in res['ats_issues']]) if res['ats_issues'] else 'No major ATS formatting issues.'}
==================================================
"""
        st.download_button(
            label="📥 Download Full Evaluation Report (.txt)",
            data=report_text,
            file_name="Resume_Analysis_Report.txt",
            mime="text/plain"
        )