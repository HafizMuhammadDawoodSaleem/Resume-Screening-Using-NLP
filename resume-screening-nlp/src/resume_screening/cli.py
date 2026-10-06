"""Batch ranking for a selected job from a jobs dataset."""
import argparse
from pathlib import Path

import pandas as pd

from .documents import load_jobs, load_resumes
from .ranking import DEFAULT_MODEL, KeywordEncoder, SemanticEncoder, rank_resumes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumes", default="data/resumes.csv")
    parser.add_argument("--jobs", default="data/jobs.csv")
    parser.add_argument("--job-id", default="J001")
    parser.add_argument("--resume-text-column", default="resume_text")
    parser.add_argument("--resume-id-column", default="id")
    parser.add_argument("--resume-name-column", default="name")
    parser.add_argument("--job-text-column", default="description")
    parser.add_argument("--job-id-column", default="id")
    parser.add_argument("--job-title-column", default="title")
    parser.add_argument("--backend", choices=["semantic", "tfidf"], default="semantic")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="outputs/rankings.csv")
    args = parser.parse_args()
    try:
        resumes = load_resumes(args.resumes, args.resume_text_column, args.resume_id_column,
                               args.resume_name_column)
        jobs = load_jobs(args.jobs, args.job_text_column, args.job_id_column, args.job_title_column)
        selected = jobs.loc[jobs.id == args.job_id]
        if selected.empty:
            raise ValueError(f"Unknown job ID: {args.job_id}. Available: {', '.join(jobs.id)}")
        encoder = SemanticEncoder(args.model) if args.backend == "semantic" else KeywordEncoder()
        result = rank_resumes(selected.iloc[0].text, resumes, encoder, args.top_k)
        frame = pd.DataFrame([match.to_dict() for match in result])
        frame.insert(0, "job_id", args.job_id)
        frame.insert(1, "backend", args.backend)
        frame.insert(2, "model", args.model if args.backend == "semantic" else "TF-IDF")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(frame[["rank", "candidate", "cosine_similarity"]].to_string(index=False))
        print(f"\nSaved: {output.resolve()}")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
