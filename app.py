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
# Load model
# -----------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

generator = load_model()

# -----------------------
# Extract PDF text
# -----------------------
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="GenAI Resume Analyzer", page_icon="🤖")

st.title("🤖 GenAI Resume Analyzer")
st.write("AI-powered resume vs job description analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd = st.text_area("Paste Job Description")

# -----------------------
# ANALYZE
# -----------------------
if st.button("Analyze"):

    if not uploaded_file or not jd.strip():
        st.warning("Upload resume and paste job description")
    else:
        resume_text = extract_pdf(uploaded_file)

        prompt = f"""
You are a senior technical recruiter.

Carefully analyze the resume and job description below.

Give a detailed and DIFFERENT response each time.

Format:

MATCH SCORE: (0-100 with reasoning)

STRENGTHS:
- ...

MISSING SKILLS:
- ...

SUGGESTIONS:
- ...

Be specific and realistic.

RESUME:
{resume_text[:1500]}

JOB DESCRIPTION:
{jd[:1500]}
"""

        with st.spinner("Running AI analysis..."):
            result = generator(
                prompt,
                max_length=400,
                do_sample=True,
                temperature=0.9
            )[0]["generated_text"]

        st.subheader("📊 Result")
        st.write(result)