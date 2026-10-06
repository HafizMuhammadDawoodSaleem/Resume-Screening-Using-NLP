# Resume Screening Using NLP

Rank resumes against a job description using **Sentence Transformers embeddings** and **cosine similarity**, then inspect a relevant resume excerpt for each result.

Includes a Streamlit upload interface, batch command line tool, fictional sample datasets, and automated tests. This is an educational, locally runnable project—not a validated automated hiring system.

## Features

- Semantic comparison using `sentence-transformers/all-MiniLM-L6-v2`.
- Text normalization that preserves technical punctuation, case and negation.
- Overlapping, token-aware chunks to cover long documents.
- Ranked results with raw cosine scores and short, evidence-based explanations.
- Upload one or more **TXT, PDF or DOCX** resumes.
- Paste a job description, select a sample role, or upload a jobs CSV.
- Compare resume CSV datasets and export results to CSV.
- Explicit TF-IDF baseline for a quick offline demo.

**Bonus scope:** the upload/results front end is implemented. Named-entity extraction for skills or experience is intentionally excluded.

## Requirements

- Python **3.10–3.12** recommended; Python 3.12 was used for local validation.
- Internet access to install dependencies and download the embedding model on its first use.
- A CPU is sufficient. No API key, database, Kaggle account or GPU is needed for the bundled demo.

## Quick start

Download and extract the project ZIP, then open a terminal in the `resume-screening-nlp` folder.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

On **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

On **Windows Command Prompt**:

```bat
.venv\Scripts\activate.bat
```

On **macOS or Linux**:

```bash
source .venv/bin/activate
```

If your system uses `python3`, substitute it for `python` when creating the environment. If PowerShell blocks activation, use Command Prompt or call `.\.venv\Scripts\python.exe` directly in the following commands.

### 2. Install

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The second command installs the project and every dependency listed in `requirements.txt`. PyTorch and model-related packages can be large. Optional Linux CPU-only preparation, before installing the project:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Launch the front end

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** if the browser does not open automatically. Stop the server with **Ctrl+C**.

### 4. Run a comparison

1. Choose **Semantic embeddings** in the sidebar.
2. Leave **Sample jobs** selected and choose **Junior Data Analyst**.
3. Select **Sample resumes**, or upload files from `data/resumes/`.
4. Click **Compare resumes →**. Allow time for the initial model download.
5. Review ranked candidates, open the closest excerpts, and download results.

For a demo without downloading model weights, select **TF-IDF · offline baseline**. Dependencies still need to be installed first. Switching methods changes the scoring algorithm; TF-IDF scores are not semantic-model scores.

## Command line

Run from the project root after installation:

```bash
python -m resume_screening.cli --job-id J001 --top-k 5
```

The default backend is semantic embeddings. Results are written to `outputs/rankings.csv`.

Offline keyword demo:

```bash
python -m resume_screening.cli --backend tfidf --job-id J001 --output outputs/demo.csv
```

Custom datasets:

```bash
python -m resume_screening.cli --resumes data/private/resumes.csv --jobs data/private/jobs.csv --job-id J001 --top-k 10 --output outputs/custom.csv
```

To use a pre-downloaded compatible Sentence Transformer model from disk, add `--model /path/to/model`. Run `python -m resume_screening.cli --help` for all options.

## Datasets

The assignment recommends a resume dataset **and** a job dataset from Kaggle but does not name particular datasets. Both input types are supported. The bundled files contain five fictional resumes and three fictional jobs written for this project; they are not downloaded Kaggle data or evidence of real-world accuracy.

| File | Required columns | Purpose |
| --- | --- | --- |
| `data/resumes.csv` | `id`, `name`, `resume_text` | Candidate documents |
| `data/jobs.csv` | `id`, `title`, `description` | Role descriptions |

IDs must be unique within each file. Required cells must not be blank. Use UTF-8 CSV; quote fields containing commas or newlines. Text is limited to 100,000 characters per record. Avoid including unnecessary personal information.

### Use Kaggle or other CSV files

1. Download both datasets manually and review their licenses.
2. Keep private or third-party data in `data/private/`, which is excluded by `.gitignore`.
3. Normalize the columns to the schema above, or supply column mappings through the CLI.
4. Add stable unique IDs and a display-name column when the dataset lacks them. Do not use category labels as unique IDs.

For a resume CSV with columns `ID`, `Category`, and `Resume_str`:

```bash
python -m resume_screening.cli --resumes data/private/Resume.csv --resume-id-column ID --resume-name-column Category --resume-text-column Resume_str --jobs data/jobs.csv --job-id J001
```

Here `Category` is only a display label; `ID` identifies each candidate. Job mappings are available as `--job-id-column`, `--job-title-column`, and `--job-text-column`. The front-end CSV uploads use the standard schema only.

## How matching works

1. **Read:** extract text from TXT, text-based PDF or DOCX, or load CSV records with pandas.
2. **Normalize:** apply Unicode normalization and collapse whitespace. Stop words are not removed, and negation is retained.
3. **Chunk:** split each document into overlapping model-token windows within the model's input limit.
4. **Embed:** encode the chunks using Sentence Transformers and average their embeddings into one vector per document.
5. **Rank:** use scikit-learn cosine similarity between the job vector and each resume vector, then sort descending. Ties preserve input order.
6. **Explain:** compare short resume excerpts to the job vector and display the closest passage with its own score.

Cosine similarity is `dot(job, resume) / (norm(job) × norm(resume))`. It ranges from **−1 to 1**. A score of `0.85` is textual similarity—not an 85% chance of success or proof that 85% of requirements are met. There are no automatic accept/reject thresholds.

The explanation is a deterministic summary plus an actual normalized-text excerpt. It is not generated by an LLM and does not verify qualifications. Whole-document scores and excerpt scores differ because they compare different text representations.

The optional TF-IDF baseline fits a shared word/unigram/bigram vocabulary over the comparison inputs and uses the same cosine-ranking flow. Its scores can change with the candidate pool. It does not require model weights and is never silently substituted for the semantic model.

## Project structure

```text
resume-screening-nlp/
├── app.py                         # Streamlit bonus front end
├── README.md
├── VALIDATION.md                   # Checks performed for this deliverable
├── requirements.txt
├── pyproject.toml                  # Installable Python package
├── .gitignore
├── .streamlit/config.toml
├── data/
│   ├── resumes.csv                 # Fictional resume dataset
│   ├── jobs.csv                    # Fictional job dataset
│   └── resumes/                    # Sample TXT uploads
├── src/resume_screening/
│   ├── __init__.py
│   ├── documents.py                # Extraction and CSV validation
│   ├── ranking.py                  # Embeddings, ranking and evidence
│   └── cli.py                      # Batch interface
└── tests/
    ├── test_project.py             # Core ranking and parser tests
    └── test_ui.py                  # Front-end sample comparison
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The fast suite does not download a model. It checks known cosine rankings, ties, evidence selection, long-document coverage, CSV validation, document extraction, preprocessing, and invalid inputs. The semantic unit checks use controlled vectors and a tokenizer stub; see `VALIDATION.md` for the actual end-to-end checks and any remaining limitations.

For a manual semantic smoke test, run the default CLI command above and confirm it completes with finite scores and relevant evidence. Meaningful quality evaluation additionally requires representative labeled resume/job pairs, recruiter review, and ranking metrics such as Precision@k or NDCG; no accuracy claim is made from the fictional examples.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named resume_screening` | Activate the environment and run `python -m pip install -e .` from the project root. |
| Model download fails | Check access to Hugging Face, or explicitly select TF-IDF for the demo. The default model requires a first-time download. |
| Missing `socksio` behind a SOCKS proxy | Install the optional proxy support with `python -m pip install "httpx[socks]"`. |
| A PDF produces no text | It may be an image scan. OCR it first; OCR is outside this project's scope. |
| File cannot be read | Use a valid, unencrypted PDF, DOCX, or UTF-8 TXT file. Legacy `.doc` is unsupported. |
| Upload is too large | Each file must be at most 5 MB. PDFs are limited to 50 pages. |
| CSV is rejected | Check headers, unique IDs, required nonblank fields and CSV quoting. |
| Port 8501 is occupied | Run `python -m streamlit run app.py --server.port 8502`. |
| Long startup or install | Model libraries can be large. Use a supported Python environment and allow installation/download to finish. |

## Scope and limitations

- Uploads are processed in server memory and are not intentionally written to disk. Only model resources are globally cached. Running on another server sends documents to that server; the default configuration binds to localhost.
- This prototype has no authentication, access controls, OCR, database or production deployment configuration.
- PDF reading order and complex DOCX layouts may affect extraction. Headers, footers, embedded images and text boxes are not guaranteed to be read.
- The default model is primarily suited to English. Averaged chunks can dilute important requirements, and similarity can miss negation, required credentials or years of experience.
- Names and personal details are not automatically redacted. Ranking may reflect irrelevant text or model bias. Use human review; never treat the result as an autonomous employment decision.
- Front-end limits: 50 file uploads or 200 resume CSV rows per comparison. The CLI supports larger datasets but computes in memory and is intended for modest batches.
- Dependency ranges are provided, not a fully locked deployment environment. See validation notes for what was actually exercised.

## Put it on GitHub

Create an empty repository in GitHub, then run these commands from this project folder. Replace the example URL with your repository URL:

```bash
git init
git add .
git commit -m "Add NLP resume screening project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/resume-screening-nlp.git
git push -u origin main
```

Before committing, use `git status` to confirm that no real resumes, private data, environment files or generated outputs are included. Choose a license before public redistribution; no license has been selected on your behalf.

## References

- [Sentence Transformers: semantic textual similarity](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
- [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

Built for **Task 8: Resume Screening Using NLP**.
