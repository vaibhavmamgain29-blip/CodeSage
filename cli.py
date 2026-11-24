import argparse
import sys
from pathlib import Path
from tabulate import tabulate
from codesage.indexing.indexer import RepoIndexer
from codesage.retrieval.hybrid import HybridRetriever
from codesage.qa.service import QAService
from codesage.review.service import ReviewService, fetch_github_pr_diff
from codesage.config import settings


def main():
    parser = argparse.ArgumentParser(prog="codesage", description="CodeSage CLI: AI Code Review & Repo Q&A")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("repo", nargs="?", default=".", help="Local path or git URL of repository (default: .)")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed repository")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid", help="Retrieval mode")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    search_parser.add_argument("--repo", default=".", help="Repository path")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the repository")
    ask_parser.add_argument("question", help="Question text")
    ask_parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid", help="Retrieval mode")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks")
    ask_parser.add_argument("--repo", default=".", help="Repository path")

    # Review command
    review_parser = subparsers.add_parser("review", help="Review a PR or diff")
    review_parser.add_argument("target", help="Unified diff file path or GitHub PR URL")
    review_parser.add_argument("--severity", choices=["low", "medium", "high"], default=None, help="Filter by min severity")
    review_parser.add_argument("--min-confidence", type=float, default=0.0, help="Min confidence threshold (0.0 to 1.0)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "index":
        indexer = RepoIndexer(args.repo)
        print(f"Indexing repository at '{args.repo}'...")
        stats = indexer.index(progress_callback=lambda p, msg: print(f"[{int(p*100)}%] {msg}"))
        print("\nIndexing complete!")
        print(f"Files: {stats['files_indexed']} | Chunks: {stats['chunks_created']} | Elapsed: {stats['elapsed_sec']}s")

    elif args.command == "search":
        indexer = RepoIndexer(args.repo)
        if not indexer.load_existing_index():
            print("No existing index found, indexing now...")
            indexer.index()

        retriever = HybridRetriever(indexer.vector_store, indexer.bm25_index)
        results = retriever.retrieve(args.query, mode=args.mode, top_k=args.top_k)

        rows = []
        for rank, (chunk, score) in enumerate(results, start=1):
            rows.append([rank, chunk.file_path, f"{chunk.start_line}-{chunk.end_line}", chunk.symbol_name, f"{score:.4f}"])

        print(f"\n--- Top {len(results)} Results for '{args.query}' (Mode: {args.mode.upper()}) ---")
        print(tabulate(rows, headers=["#", "File", "Lines", "Symbol", "Score"], tablefmt="github"))

    elif args.command == "ask":
        indexer = RepoIndexer(args.repo)
        if not indexer.load_existing_index():
            print("No existing index found, indexing now...")
            indexer.index()

        retriever = HybridRetriever(indexer.vector_store, indexer.bm25_index)
        qa = QAService(retriever)
        resp = qa.ask(args.question, mode=args.mode, top_k=args.top_k)

        print("\n" + "="*50)
        print(f"Question: {resp.question}")
        print("="*50)
        print(f"\nAnswer:\n{resp.answer}")

        if resp.citations:
            print("\nCitations:")
            for c in resp.citations:
                status = "VALID" if c.is_valid else "UNVERIFIED"
                print(f"- [{status}] {c.raw_citation}")

    elif args.command == "review":
        target = args.target
        if target.startswith("http://") or target.startswith("https://"):
            print(f"Fetching GitHub PR diff from: {target}...")
            diff_text = fetch_github_pr_diff(target)
        else:
            p = Path(target)
            if not p.exists():
                print(f"Error: Diff file '{target}' not found.")
                sys.exit(1)
            diff_text = p.read_text(encoding="utf-8")

        reviewer = ReviewService()
        print("Running code review...")
        review = reviewer.review_diff(diff_text, severity_filter=args.severity, min_confidence=args.min_confidence)

        print("\n" + "="*50)
        print(f"Summary: {review.summary}")
        print("="*50)
        print(f"Findings ({len(review.findings)}):")
        for idx, f in enumerate(review.findings, start=1):
            print(f"\n[{idx}] [{f.severity.value.upper()}] {f.category.value.upper()} | {f.file}:{f.line} (Confidence: {int(f.confidence*100)}%)")
            print(f"Issue: {f.issue}")
            print(f"Suggestion: {f.suggestion}")


if __name__ == "__main__":
    main()
