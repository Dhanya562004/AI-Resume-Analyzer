# AI Resume Analyzer

A small GenAI app: paste a resume and a job description, get a match score,
missing skills, and improvement suggestions from an LLM.

## Setup (run this on your own machine, not in this sandbox)

1. Get an OpenAI API key: https://platform.openai.com/api-keys
   (You'll need to add a small amount of billing credit — a few dollars covers
   hundreds of runs with gpt-4o-mini, it's cheap.)

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your API key as an environment variable — do NOT paste it into the code.

   On Mac/Linux:
   ```
   export OPENAI_API_KEY="your-key-here"
   ```
   On Windows (PowerShell):
   ```
   $env:OPENAI_API_KEY="your-key-here"
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

5. It opens in your browser at `http://localhost:8501`. Paste your real resume
   and a real job description and try it.

## Before your interview — understand these, don't just memorize them

You need to be able to answer follow-ups on your own project. Here's what to
actually understand, in plain terms:

**"Why environment variable instead of just putting the key in the code?"**
If the key is hardcoded and you ever push this to GitHub (even a private repo
that becomes public later, or a screen-share), anyone can see it and use it —
on your bill. Environment variables keep secrets out of the source code.

**"What does temperature=0.3 do?"**
Temperature controls randomness in the model's output. Low temperature (like
0.3) makes responses more consistent and focused — good for something like
scoring a resume, where you want reliable output, not creative variation.
High temperature (closer to 1) makes output more varied/creative — better for
things like brainstorming or story writing.

**"Why did you structure the prompt with numbered sections?"**
Without structure, the model might respond in a different format every time,
which is hard to parse or display consistently. Telling it exactly what
sections to return makes the output predictable.

**"What would you improve if you had more time?"**
Honest answers here are good, not a weakness: comparing multiple resume
versions at once, extracting structured data (JSON) instead of free text so
it could be rendered as a proper UI instead of raw text, or letting the user
upload a PDF instead of pasting text.

**"Did you write this yourself?"**
Yes — say so plainly, and mean it. You should understand every part of this
file well enough to change it live if someone asks you to add a small
feature. If you can't explain a line, don't claim it — go back over it until
you can.

## Optional 20-minute extensions (only if you actually build them)

- Accept PDF upload instead of pasted text (use `pypdf` to extract text)
- Store past analyses in a list so the user can compare multiple job
  descriptions against the same resume
- Add a simple bar chart of match score if you run it against several JDs
