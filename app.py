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
from transformers import pipeline
import PyPDF2

# -----------------------
# Load LLM (FREE model)
# -----------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

# -----------------------
# Extract text from PDF
# -----------------------
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -----------------------
# Page UI
# -----------------------
st.set_page_config(page_title="GenAI Resume Analyzer", page_icon="🤖")

st.title("🤖 GenAI Resume Analyzer")
st.caption("Upload your resume and compare with a job description using AI")

# -----------------------
# Inputs
# -----------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd = st.text_area("Paste Job Description")

# -----------------------
# Analyze
# -----------------------
if st.button("Analyze", type="primary"):

    if not uploaded_file or not jd.strip():
        st.warning("Please upload resume and paste job description")
    else:
        resume_text = extract_pdf(uploaded_file)

        prompt = f"""
You are an expert technical recruiter.

Analyze the following resume and job description.

Give output in this format:

1. Match Score (0-100%) with reason
2. Key Strengths (bullet points)
3. Missing Skills (bullet points)
4. Suggestions to Improve (clear, human advice)

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{jd[:2000]}
"""

        with st.spinner("Analyzing with AI..."):
            result = generator(prompt, max_length=512, do_sample=True)[0]["generated_text"]

        st.markdown("## 📊 Analysis Result")
        st.write(result)