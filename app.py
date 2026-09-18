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
import requests
import PyPDF2

# -----------------------
# Load API key
# -----------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# -----------------------
# Extract PDF
# -----------------------
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# -----------------------
# Call API
# -----------------------
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="GenAI Resume Analyzer", page_icon="🤖")

st.title("🤖 GenAI Resume Analyzer")
st.write("AI-powered resume analyzer")

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
You are a professional recruiter.

Analyze resume vs job description.

Give:
1. Match Score (0-100 + reason)
2. Strengths
3. Missing Skills
4. Suggestions

RESUME:
{resume_text[:1500]}

JOB DESCRIPTION:
{jd[:1500]}
"""

        with st.spinner("Analyzing..."):
            result = query({"inputs": prompt})

        try:
            output = result[0]["generated_text"]
        except:
            output = str(result)

        st.subheader("📊 Result")
        st.write(output)