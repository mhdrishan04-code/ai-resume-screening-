# AI Resume Screening System

A responsive Streamlit app that compares PDF, DOCX, and TXT resumes with a job description using TF-IDF cosine similarity and transparent skill matching.

## Features
- Multiple resume uploads
- PDF, DOCX, and TXT text extraction
- Job-description matching using TF-IDF and cosine similarity
- Skill coverage and potential skill-gap analysis
- Ranked candidate table and CSV export
- Responsive Streamlit interface for desktop and mobile browsers

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud
1. Push this repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose the repository, branch `main`, and main file path `app.py`.
4. Deploy. Streamlit provides a public `*.streamlit.app` URL.

## Scoring notes
The displayed relevance score combines TF-IDF text similarity (65%) and required-skill coverage (35%) when skills are detected in the job description. If no skills are detected, the score uses text similarity alone. This is a transparent heuristic, not a trained or validated prediction model. It must not be the sole basis for hiring decisions.

## Privacy and limitations
- Resume contents can contain sensitive personal information. Upload only documents you are authorized to process.
- Files are processed in the current app session; the app does not intentionally save resumes to a database.
- Scanned/image-only PDFs may require OCR, which is not included.
- The skill list is a starter dictionary and should be reviewed and expanded for the target domain.
- Human review is required; do not use protected attributes or infer candidate quality from this score alone.
