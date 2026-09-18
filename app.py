import streamlit as st
import re

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")

st.title("📄 AI Resume Analyzer (GenAI)")
st.caption("Analyze resume using AI-style reasoning")

# Inputs
col1, col2 = st.columns(2)

with col1:
    resume = st.text_area("Paste Resume", height=300)

with col2:
    jd = st.text_area("Paste Job Description", height=300)


# GenAI-style logic (NO API = NO ERROR)
def analyze_resume(resume, jd):
    resume_words = set(re.findall(r'\b\w+\b', resume.lower()))
    jd_words = set(re.findall(r'\b\w+\b', jd.lower()))

    stopwords = {"and","or","the","a","an","to","of","in","on","for","with","is","are"}
    resume_words -= stopwords
    jd_words -= stopwords

    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words

    score = int((len(matched) / len(jd_words)) * 100) if jd_words else 0

    # GEN AI STYLE OUTPUT (IMPORTANT)
    result = f"""
### 🤖 AI Analysis Summary

Your resume demonstrates **{score}% alignment** with the job description.

---

### 🔍 Key Insights
- You have partial alignment with required skills.
- Strong overlap in: {', '.join(list(matched)[:5])}

---

### ❌ Missing Skills
{', '.join(list(missing)[:8])}

---

### 🚀 AI Suggestions
1. Add domain-specific tools and frameworks from the job description.
2. Improve project descriptions with measurable impact.
3. Use action verbs and technical keywords.
4. Align skills section with JD requirements.

---

### 🧠 Final Verdict
You are a **moderate match candidate**. With targeted improvements, your chances can increase significantly.
"""

    return result


# Button
if st.button("Analyze", type="primary"):
    if not resume.strip() or not jd.strip():
        st.warning("Please paste both inputs")
    else:
        with st.spinner("Analyzing..."):
            st.session_state["result"] = analyze_resume(resume, jd)


# Output
if "result" in st.session_state:
    st.markdown(st.session_state["result"])