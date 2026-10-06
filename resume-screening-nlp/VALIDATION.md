# Validation notes

Validated on Python 3.12.14 in the supplied execution environment.

| Check | Result |
| --- | --- |
| Editable package installation | Passed |
| Python syntax compilation | Passed |
| Core automated tests | 9 passed |
| Streamlit AppTest sample comparison | Passed; 5 result cards rendered in TF-IDF mode |
| Full automated suite | 10 passed |
| TF-IDF command line run with both sample CSV datasets | Passed; exported ranked results |
| Valid text-based PDF extraction | Passed using a temporary generated PDF |
| TXT and DOCX extraction, including DOCX tables | Passed |
| Actual pretrained semantic-model inference | Not completed; initial model download stalled and the check was interrupted |

The semantic implementation's vector ranking and token chunking are tested with controlled fixtures. These tests do not replace a run with downloaded model weights. No semantic scores or model-accuracy results are claimed from this environment.

Installed direct dependency versions used during checks:

```text
numpy                 2.3.5
pandas                2.2.3
scikit-learn          1.8.0
sentence-transformers 5.7.0
streamlit             1.64.0
pypdf                 6.10.0
python-docx           1.2.0
```

The network environment initially required optional SOCKS proxy support. After installing it, the model-weight download did not complete. The README includes proxy troubleshooting and the explicit offline baseline. The package does not silently replace semantic matching when a model load fails.

The UI was exercised using Streamlit's official AppTest runner, not a visual browser screenshot test. Windows/macOS installation and real-world ranking quality were not validated here. Model weights and third-party datasets are not bundled in the ZIP.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m resume_screening.cli --backend tfidf --job-id J001
python -m resume_screening.cli --job-id J001 --output outputs/semantic.csv
python -m streamlit run app.py
```

The semantic command requires a successful first-time model download. All sample data is fictional.
