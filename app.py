import io
import re
from typing import List, Dict

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

SKILL_LIBRARY = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "mysql",
    "postgresql", "mongodb", "html", "css", "react", "node.js", "fastapi", "flask",
    "django", "streamlit", "machine learning", "deep learning", "natural language processing",
    "nlp", "scikit-learn", "pandas", "numpy", "tensorflow", "pytorch", "data analysis",
    "power bi", "tableau", "excel", "aws", "azure", "gcp", "docker", "kubernetes",
    "git", "github", "linux", "communication", "project management", "leadership",
    "data visualization", "statistics", "computer vision", "rest api", "api", "agile",
    "cybersecurity", "networking", "software testing", "selenium", "communication skills"
]

def extract_pdf(data: bytes) -> str:
    """Extract embedded PDF text, using OCR for scanned/image-only pages."""
    text = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text().strip()
            if len(page_text) < 20:
                # Scanned PDFs often contain images rather than a searchable text layer.
                try:
                    ocr_page = page.get_textpage_ocr(language="eng", dpi=200, full=True)
                    page_text = page.get_text(textpage=ocr_page).strip()
                except Exception:
                    # Keep the app usable even if OCR language data is unavailable.
                    pass
            text.append(page_text)
    return "\n".join(text).strip()

def extract_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs).strip()

def extract_text(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf(data)
    if name.endswith(".docx"):
        return extract_docx(data)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def find_skills(text: str) -> List[str]:
    t = normalize(text)
    found = []
    for skill in SKILL_LIBRARY:
        pattern = r"(?<![a-z0-9+#.])" + re.escape(skill.lower()) + r"(?![a-z0-9+#.])"
        if re.search(pattern, t):
            found.append(skill)
    return sorted(set(found))

def score_resume(resume_text: str, job_text: str) -> Dict:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
    try:
        matrix = vectorizer.fit_transform([job_text, resume_text])
        similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)
    except ValueError:
        similarity = 0.0

    job_skills = find_skills(job_text)
    resume_skills = find_skills(resume_text)
    matched = sorted(set(job_skills) & set(resume_skills))
    missing = sorted(set(job_skills) - set(resume_skills))
    skill_score = (len(matched) / len(job_skills) * 100) if job_skills else 0.0

    # Transparent weighted relevance score; not a hiring decision or validated prediction.
    overall = (0.65 * similarity + 0.35 * skill_score) if job_skills else similarity
    return {
        "match_score": round(overall, 1),
        "text_similarity": round(similarity, 1),
        "skill_coverage": round(skill_score, 1),
        "matched_skills": ", ".join(matched) if matched else "None detected",
        "missing_skills": ", ".join(missing) if missing else ("No listed skills missing" if job_skills else "No skills detected in job description"),
        "resume_skills": ", ".join(resume_skills) if resume_skills else "None detected",
        "job_skills": job_skills,
        "matched_list": matched,
        "missing_list": missing,
    }

def main():
    st.title("📄 AI Resume Screening System")
    st.caption("NLP-powered resume relevance analysis • Streamlit • Works on desktop and mobile browsers")

    with st.sidebar:
        st.header("About")
        st.write("Upload resumes and compare them with a job description. Results are decision-support signals, not automatic hiring decisions.")
        st.info("Privacy: files are processed in this app session. Do not upload resumes unless you are authorized to process them.")
        st.markdown("**Supported files:** PDF, DOCX, TXT")

    st.subheader("1. Job requirements")
    job_title = st.text_input("Job title", placeholder="e.g., Junior Machine Learning Engineer")
    job_description = st.text_area(
        "Paste the job description and required qualifications",
        height=200,
        placeholder="Describe responsibilities, required skills, education, and experience..."
    )
    st.subheader("2. Upload candidate resumes")
    uploads = st.file_uploader(
        "Select one or more resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Scanned PDFs without a text layer may need OCR before they can be analyzed."
    )

    run = st.button("Analyze and rank resumes", type="primary", use_container_width=True)
    if run:
        if not job_description.strip():
            st.error("Please enter a job description before analysis.")
            return
        if not uploads:
            st.error("Please upload at least one resume.")
            return

        results = []
        progress = st.progress(0)
        for i, file in enumerate(uploads):
            try:
                text = extract_text(file)
                if not text:
                    results.append({"candidate_file": file.name, "error": "No readable text extracted."})
                else:
                    score = score_resume(text, job_description)
                    results.append({
                        "candidate_file": file.name,
                        "match_score": score["match_score"],
                        "text_similarity": score["text_similarity"],
                        "skill_coverage": score["skill_coverage"],
                        "matched_skills": score["matched_skills"],
                        "missing_skills": score["missing_skills"],
                        "resume_skills": score["resume_skills"],
                        "_details": score,
                        "_text": text
                    })
            except Exception as e:
                results.append({"candidate_file": file.name, "error": str(e)})
            progress.progress((i + 1) / len(uploads))

        valid = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]
        if valid:
            valid.sort(key=lambda x: x["match_score"], reverse=True)
            st.subheader("3. Candidate ranking")
            a, b, c = st.columns(3)
            a.metric("Resumes analyzed", len(valid))
            b.metric("Highest relevance score", f'{valid[0]["match_score"]:.1f}%')
            c.metric("Job title", job_title.strip() or "Not specified")

            table = pd.DataFrame([{
                "Rank": i + 1,
                "Candidate file": r["candidate_file"],
                "Overall match": r["match_score"],
                "Text similarity": r["text_similarity"],
                "Skill coverage": r["skill_coverage"],
                "Matched skills": r["matched_skills"],
                "Potential skill gaps": r["missing_skills"],
            } for i, r in enumerate(valid)])
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download ranking as CSV",
                data=table.to_csv(index=False).encode("utf-8-sig"),
                file_name="resume_screening_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.subheader("4. Candidate details")
            for i, r in enumerate(valid):
                with st.expander(f'#{i+1} — {r["candidate_file"]} — {r["match_score"]:.1f}% match'):
                    left, right = st.columns(2)
                    left.metric("Text similarity", f'{r["text_similarity"]:.1f}%')
                    right.metric("Required-skill coverage", f'{r["skill_coverage"]:.1f}%')
                    st.write("**Detected skills:**", r["resume_skills"])
                    st.write("**Skills matched:**", r["matched_skills"])
                    st.write("**Potential gaps:**", r["missing_skills"])
                    st.caption("Scores reflect textual similarity and a simple skill dictionary. They are not validated predictions of job performance.")
        if errors:
            st.warning("Some files could not be analyzed:")
            for item in errors:
                st.write(f'- **{item["candidate_file"]}:** {item["error"]}')

    st.divider()
    st.caption("Responsible use: review candidates individually, verify extracted information, and avoid using protected characteristics or this score as the sole basis for employment decisions.")

if __name__ == "__main__":
    main()
