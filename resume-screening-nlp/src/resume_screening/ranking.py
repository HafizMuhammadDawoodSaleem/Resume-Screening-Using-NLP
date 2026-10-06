"""Transparent document-level cosine ranking with supporting text excerpts."""
from dataclasses import asdict, dataclass
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .documents import Resume, validate_text

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Match:
    rank: int
    resume_id: str
    candidate: str
    cosine_similarity: float
    evidence: str
    evidence_cosine: float
    explanation: str

    def to_dict(self):
        return asdict(self)


class SemanticEncoder:
    """Token-aware, overlapping chunks avoid dropping the tail of a resume."""
    def __init__(self, model_name=DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu", trust_remote_code=False)

    def chunk(self, text):
        tokenizer = self.model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        limit = max(16, min(240, self.model.max_seq_length - 8))
        stride = max(1, limit - 32)
        chunks = []
        for start in range(0, len(tokens), stride):
            chunks.append(tokenizer.decode(tokens[start:start + limit], skip_special_tokens=True))
            if start + limit >= len(tokens):
                break
        return chunks

    def encode(self, texts):
        return self.model.encode(texts, batch_size=32, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)


class KeywordEncoder:
    """Explicit offline baseline, not a substitute for semantic embeddings."""
    def chunk(self, text):
        words = text.split()
        chunks = []
        for start in range(0, len(words), 100):
            chunks.append(" ".join(words[start:start + 120]))
            if start + 120 >= len(words):
                break
        return chunks

    def encode(self, texts):
        try:
            return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
        except ValueError as exc:
            raise ValueError("Not enough readable words to compare these documents.") from exc


def evidence_chunks(text: str):
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
    # Bound long sentences too; evidence is always an actual input substring.
    return [" ".join(words[i:i + 65]) for part in parts
            if (words := part.split()) for i in range(0, len(words), 65)]


def rank_resumes(job: str, resumes: list[Resume], encoder, top_k: int | None = None) -> list[Match]:
    job = validate_text(job)
    if not resumes:
        raise ValueError("Add at least one resume.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if len({r.id for r in resumes}) != len(resumes):
        raise ValueError("Resume IDs must be unique.")
    docs = [job] + [validate_text(r.text) for r in resumes]
    groups = [encoder.chunk(text) for text in docs]
    excerpts = [evidence_chunks(text) for text in docs[1:]]
    all_text = [chunk for group in groups + excerpts for chunk in group]
    vectors = encoder.encode(all_text)
    offsets = np.cumsum([0] + [len(group) for group in groups + excerpts])
    # Averaging all chunks covers the full document; sklearn normalizes for cosine.
    document_vectors = np.vstack([
        np.asarray(vectors[offsets[i]:offsets[i + 1]].mean(axis=0)).reshape(-1)
        for i in range(len(groups))
    ])
    scores = cosine_similarity(document_vectors[1:], document_vectors[:1]).ravel()
    order = sorted(range(len(resumes)), key=lambda i: (-float(scores[i]), i))
    if top_k is not None:
        order = order[:top_k]
    results = []
    for position, i in enumerate(order, 1):
        group_index = len(groups) + i
        similarities = cosine_similarity(
            vectors[offsets[group_index]:offsets[group_index + 1]], document_vectors[:1]
        ).ravel()
        best = int(np.argmax(similarities))
        score = float(np.clip(scores[i], -1, 1))
        evidence_score = float(np.clip(similarities[best], -1, 1))
        explanation = (
            f"Ranked #{position} by whole-document cosine similarity ({score:.3f}). "
            "The excerpt shown is this resume's closest passage to the job description; "
            "it is supporting text, not verification that all requirements are met."
        )
        results.append(Match(position, resumes[i].id, resumes[i].name, score,
                             excerpts[i][best], evidence_score, explanation))
    return results
