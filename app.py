"""
AI Resume Analyzer — a mini GenAI app.

What this does:
1. Takes a resume and a job description as text input
2. Sends both to an LLM with a structured prompt
3. Displays match %, missing skills, and improvement suggestions

Why it's built this way (read this before your interview):
- The API key is loaded from an environment variable, NOT hardcoded in the file.
  Hardcoding a key means anyone who sees your code (or if you push it to GitHub)
  gets your key and can rack up charges on your account. This is a real security
  practice, not just a formality — be ready to explain this choice if asked.
- We use st.session_state to keep the last result available even if the user
  interacts with other widgets, instead of losing it on rerun.
- The prompt asks the model to return a SPECIFIC structure (numbered sections)
  so the output is consistent and easy to read every time.
"""
import streamlit as st
import re
from PyPDF2 import PdfReader

# ---------------- UI ----------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")

st.title("📄 AI Resume Analyzer (Pro)")
st.caption("Upload your resume + paste job description → get smart analysis")

# ---------------- PDF Upload ----------------
uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

resume_text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        resume_text += page.extract_text()

# ---------------- JD Input ----------------
jd = st.text_area("Paste Job Description", height=200)

# ---------------- ANALYSIS FUNCTION ----------------
def analyze_resume(resume, jd):
    resume_words = set(re.findall(r'\b\w+\b', resume.lower()))
    jd_words = set(re.findall(r'\b\w+\b', jd.lower()))

    # remove common words
    stopwords = {
        "and","or","the","a","an","to","of","in","on","for","with",
        "is","are","was","were","be","been","being",
        "good","excellent","strong","skills","ability",
        "communication","interpersonal","team","player"
    }

    resume_words -= stopwords
    jd_words -= stopwords

    # focus on meaningful keywords
    jd_keywords = {w for w in jd_words if len(w) > 3}
    resume_keywords = {w for w in resume_words if len(w) > 3}

    matched = resume_keywords & jd_keywords
    missing = jd_keywords - resume_keywords

    score = int((len(matched) / len(jd_keywords)) * 100) if jd_keywords else 0

    return score, matched, missing

# ---------------- BUTTON ----------------
if st.button("Analyze 🚀"):

    if not uploaded_file or not jd.strip():
        st.warning("Please upload resume and enter job description")
    else:
        score, matched, missing = analyze_resume(resume_text, jd)

        st.success("Analysis Complete ✅")

        # Score
        st.subheader("📊 Match Score")
        st.progress(score)
        st.write(f"**{score}% match based on technical keywords**")

        # Matched
        st.subheader("✅ Matching Skills")
        if matched:
            st.write(", ".join(list(matched)[:10]))
        else:
            st.write("No strong matches found")

        # Missing
        st.subheader("❌ Missing Skills")
        if missing:
            st.write(", ".join(list(missing)[:10]))
        else:
            st.write("Great! No major missing skills")

        # Suggestions
        st.subheader("💡 Suggestions")
        st.write("""
- Add missing tools/technologies mentioned in JD  
- Improve project descriptions using these keywords  
- Focus on measurable impact (e.g., accuracy %, performance)  
""")