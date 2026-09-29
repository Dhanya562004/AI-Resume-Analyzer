# 📄 AI Resume Analyzer & ATS Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-resume-analyzer-npmgmltfplr6r4ayqbfzvg.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Dhanya562004%2FAI--Resume--Analyzer-purple.svg)](https://github.com/Dhanya562004/AI-Resume-Analyzer)

A modern, GenAI-powered web application that analyzes resumes against target job descriptions. It extracts text from uploaded **PDFs or raw text**, calculates match compatibility scores using TF-IDF cosine similarity and skill extraction NLP algorithms, audits ATS compliance, highlights missing keywords, and provides actionable recommendations.

🚀 **Live Demo**: [https://ai-resume-analyzer-npmgmltfplr6r4ayqbfzvg.streamlit.app/](https://ai-resume-analyzer-npmgmltfplr6r4ayqbfzvg.streamlit.app/)

---

## 🔥 Key Features

- **📄 Multi-Format Resume Input**:
  - Direct PDF file upload with resilient multi-library extraction (`pypdf`, `PyPDF2`, `pdfplumber`).
  - Text area input mode for quick copy-pasting.
  - 1-Click sample data loader for fast demo testing.

- **📊 Comprehensive Match Scoring**:
  - **TF-IDF Semantic Cosine Similarity**: Evaluates textual similarity beyond surface-level keyword count.
  - **Domain Skill Coverage**: Checks technical skills, tools, and soft skills against a predefined taxonomy.
  - **Overall Keyword Overlap**: Identifies matched and missing job description requirements.

- **🏷️ Skill Gap & Keyword Analysis**:
  - Visual color-coded pill tags for **Matched Skills** (Green) and **Missing Skills** (Red).
  - List of top missing keywords to target for resume optimization.
  - Interactive Plotly chart displaying skill breakdown.

- **🤖 ATS Compliance & Structural Audit**:
  - Section presence check (`Summary`, `Experience`, `Education`, `Skills`, `Projects`).
  - Resume length & word count audit (300-800 word target).
  - Bullet-point density metric for ATS readability.

- **💡 Actionable AI Recommendations**:
  - Personalized improvement steps to boost match rate above 80%+.
  - Suggested action verbs to strengthen experience achievements.
  - One-click downloadable evaluation report (`.txt` format).

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Programming Language**: [Python 3.9+](https://www.python.org/)
- **PDF Extraction**: `pypdf`, `PyPDF2`, `pdfplumber`
- **Data Science & NLP**: `scikit-learn`, `pandas`, `re` (Regular Expressions)
- **Interactive Visualizations**: `plotly`

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.9 or higher installed on your system.
- Git installed.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Dhanya562004/AI-Resume-Analyzer.git
   cd AI-Resume-Analyzer
   ```

2. **Create & Activate a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

---

## 📖 How to Use

1. **Upload Resume**: Select the **📄 Upload PDF** option and upload your resume file, or paste raw text under **✏️ Paste Raw Text**.
2. **Input Job Description**: Paste the target job responsibilities and skills requirements into the Job Description field.
3. **Run Analysis**: Click **🚀 Analyze Resume & Match Job**.
4. **Explore Dashboard**:
   - View your overall match score and ATS readiness score.
   - Inspect missing skills in the **Skill & Gap Analysis** tab.
   - Check structural recommendations in **ATS Compliance Audit**.
   - Download your personalized evaluation report!

---

## 🌐 Deployment

This application is optimized for deployment on **Streamlit Community Cloud**.

1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Select your repository, set main file to `app.py`, and click **Deploy**.
4. Access the live app at: `https://ai-resume-analyzer-npmgmltfplr6r4ayqbfzvg.streamlit.app/`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Dhanya562004/AI-Resume-Analyzer/issues).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
