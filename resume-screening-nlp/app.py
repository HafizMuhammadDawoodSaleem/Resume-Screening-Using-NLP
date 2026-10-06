"""Bonus front end: upload resumes and inspect matches to a selected job."""
from pathlib import Path

import pandas as pd
import streamlit as st

from resume_screening.documents import Resume, extract_text, load_jobs, load_resumes
from resume_screening.ranking import DEFAULT_MODEL, KeywordEncoder, SemanticEncoder, rank_resumes

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Resume Match | NLP", page_icon="🔎", layout="wide")


@st.cache_resource
def semantic_model():
    # Cache model weights only, never uploaded documents or per-user results.
    return SemanticEncoder(DEFAULT_MODEL)


st.caption("NLP PROJECT / DOCUMENT SIMILARITY")
st.title("Find the experience that fits.")
st.write("Compare resumes with a job description and review the supporting evidence.")

with st.sidebar:
    st.header("Comparison settings")
    mode = st.radio("Matching method", ["Semantic embeddings", "TF-IDF · offline baseline"])
    top_k = st.slider("Results to show", 1, 20, 5)
    st.divider()
    st.caption("Semantic mode uses all-MiniLM-L6-v2. The first run downloads the model; later runs reuse its local cache.")
    st.caption("TF-IDF measures word overlap. It is provided for offline demos and is not the semantic model.")
    st.caption("Local prototype • no API key required")

left, right = st.columns([1, 1], gap="large")
resumes = []
with left:
    st.subheader("1. Choose the role")
    job_source = st.radio("Job description source", ["Sample jobs", "Paste description", "Upload jobs CSV"], horizontal=True)
    jobs = None
    if job_source == "Sample jobs":
        jobs = load_jobs(ROOT / "data/jobs.csv")
    elif job_source == "Upload jobs CSV":
        job_file = st.file_uploader("Jobs dataset", type=["csv"])
        st.caption("Columns: id, title, description")
        if job_file:
            try:
                jobs = load_jobs(job_file)
            except Exception as exc:
                st.error(f"Jobs CSV: {exc}")
    if jobs is not None:
        choice = st.selectbox("Role", jobs.id.tolist(),
                              format_func=lambda value: jobs.loc[jobs.id == value, "name"].iloc[0])
        # Include selection in widget key so changing roles updates the description.
        row = jobs.loc[jobs.id == choice].iloc[0]
        job_text = st.text_area("Job description", row.text, height=240, key=f"job-{choice}-{row.text}")
    else:
        job_text = st.text_area("Job description", height=240, placeholder="Paste responsibilities and requirements…")

with right:
    st.subheader("2. Add resumes")
    source = st.radio("Resume source", ["Upload files", "Sample resumes", "Upload resumes CSV"], horizontal=True)
    if source == "Upload files":
        uploads = st.file_uploader("Drop one or more resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        st.caption("PDF, DOCX or UTF-8 TXT • 5 MB each • up to 50 files")
        if len(uploads) > 50:
            st.error("Please upload no more than 50 resumes.")
        else:
            for index, upload in enumerate(uploads):
                try:
                    resumes.append(Resume(f"upload-{index + 1}", upload.name,
                                          extract_text(upload.name, upload.getvalue())))
                except ValueError as exc:
                    st.warning(f"Skipped {upload.name}: {exc}")
    elif source == "Sample resumes":
        resumes = load_resumes(ROOT / "data/resumes.csv")
        st.info("Five fictional resumes are ready to compare. No real candidate data is included.")
    else:
        resume_file = st.file_uploader("Resume dataset", type=["csv"])
        st.caption("Columns: id, name, resume_text • up to 200 rows in the front end")
        if resume_file:
            try:
                resumes = load_resumes(resume_file)
                if len(resumes) > 200:
                    st.error("Limit the CSV to 200 rows, or use the batch command line tool.")
                    resumes = []
            except Exception as exc:
                st.error(f"Resumes CSV: {exc}")
    st.metric("Resumes ready", len(resumes))

st.caption("Similarity measures textual alignment, not eligibility, qualification or probability of hiring. Review the original resume before making decisions.")
if st.button("Compare resumes →", type="primary", disabled=not (job_text.strip() and resumes)):
    try:
        with st.spinner("Preparing model and comparing documents…"):
            encoder = semantic_model() if mode == "Semantic embeddings" else KeywordEncoder()
            matches = rank_resumes(job_text, resumes, encoder, top_k)
        st.subheader("Ranked matches")
        st.caption(f"Method: {mode} · Cosine similarity ranges from −1 to 1; higher means closer text.")
        for match in matches:
            with st.container(border=True):
                info, score = st.columns([4, 1])
                info.subheader(f"{match.rank:02d} · {match.candidate}")
                score.metric("Cosine similarity", f"{match.cosine_similarity:.3f}")
                st.write(match.explanation)
                with st.expander("View closest resume excerpt", expanded=match.rank == 1):
                    st.text(match.evidence)
                    st.caption(f"Excerpt cosine similarity: {match.evidence_cosine:.3f}")
        frame = pd.DataFrame([match.to_dict() for match in matches])
        frame.insert(0, "method", mode)
        # Prevent text cells from being interpreted as spreadsheet formulas.
        for column in frame.select_dtypes(include="object"):
            frame[column] = frame[column].map(
                lambda value: "'" + value if value.startswith(("=", "+", "-", "@")) else value)
        st.download_button("Download results CSV", frame.to_csv(index=False), "resume-matches.csv", "text/csv")
    except Exception as exc:
        st.error(f"Comparison could not complete: {exc}")
        st.info("If the model could not download, check your connection to Hugging Face, or select the explicitly labeled TF-IDF offline baseline.")

st.divider()
st.caption("Built with Sentence Transformers, scikit-learn, pandas and Streamlit. Uploaded documents are processed in memory and are not intentionally saved by this app.")
