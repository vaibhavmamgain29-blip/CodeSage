import re
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
from codesage.review.diff_parser import parse_unified_diff, FileDiff
from codesage.review.context_builder import ContextBuilder
from codesage.review.schemas import ReviewOutput, Finding, FindingSeverity
from codesage.review.prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT
from codesage.llm.client import LLMClient
from codesage.config import settings


def fetch_github_pr_diff(pr_url: str, github_token: Optional[str] = None) -> str:
    """
    Fetch unified diff from a public GitHub PR URL.
    Supports URLs of the form: https://github.com/owner/repo/pull/123
    """
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not m:
        raise ValueError(f"Invalid GitHub PR URL: {pr_url}. Expected https://github.com/owner/repo/pull/123")

    owner, repo, pr_number = m.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"

    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "User-Agent": "CodeSage-Reviewer"
    }
    token = github_token or settings.github_token
    if token:
        headers["Authorization"] = f"token {token}"

    resp = requests.get(api_url, headers=headers, timeout=30)
    if resp.status_code == 404:
        raise ValueError(f"GitHub PR not found or repository is private: {pr_url}")
    elif resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch PR diff ({resp.status_code}): {resp.text}")

    return resp.text


class ReviewService:
    """Orchestrates diff parsing, context enrichment, and LLM-powered structured code reviews."""

    def __init__(self, repo_path: Optional[Path] = None, llm_client: Optional[LLMClient] = None):
        self.repo_path = repo_path
        self.llm = llm_client or LLMClient()
        self.context_builder = ContextBuilder(repo_path=repo_path)

    def review_diff(
        self,
        diff_text: str,
        severity_filter: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> ReviewOutput:
        """
        Conducts a structured code review over a unified diff.
        Reviews file by file to handle large diffs, merges findings, and validates with Pydantic.
        """
        file_diffs = parse_unified_diff(diff_text)
        if not file_diffs:
            return ReviewOutput(summary="No changed files or hunks detected in the provided diff.", findings=[])

        all_findings: List[Finding] = []
        file_summaries: List[str] = []

        for file_diff in file_diffs:
            # Skip deleted files or files with no hunks
            if file_diff.is_deleted or not file_diff.hunks:
                continue

            diff_context = self.context_builder.format_diff_for_prompt(file_diff)
            user_prompt = REVIEW_USER_PROMPT.format(diff_context=diff_context)

            review_dict = self._call_llm_with_validation(user_prompt)
            if review_dict:
                if "summary" in review_dict and review_dict["summary"]:
                    file_summaries.append(f"{file_diff.target_path}: {review_dict['summary']}")

                raw_findings = review_dict.get("findings", [])
                for rf in raw_findings:
                    try:
                        finding = Finding(**rf)
                        # Filter by confidence
                        if finding.confidence < min_confidence:
                            continue
                        # Filter by severity
                        if severity_filter:
                            sev_levels = {"low": 1, "medium": 2, "high": 3}
                            target_lvl = sev_levels.get(severity_filter.lower(), 1)
                            actual_lvl = sev_levels.get(finding.severity.value.lower(), 1)
                            if actual_lvl < target_lvl:
                                continue

                        all_findings.append(finding)
                    except Exception:
                        continue

        # Sort findings by severity (high -> medium -> low) and line number
        severity_rank = {FindingSeverity.HIGH: 0, FindingSeverity.MEDIUM: 1, FindingSeverity.LOW: 2}
        all_findings.sort(key=lambda f: (severity_rank.get(f.severity, 3), f.file, f.line))

        overall_summary = (
            " | ".join(file_summaries)
            if file_summaries
            else f"Reviewed {len(file_diffs)} file(s). Found {len(all_findings)} potential finding(s)."
        )

        return ReviewOutput(summary=overall_summary, findings=all_findings)

    def _call_llm_with_validation(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call LLM and retry once with a correction prompt if JSON is malformed."""
        try:
            resp = self.llm.generate(prompt=user_prompt, system_instruction=REVIEW_SYSTEM_PROMPT)
            return resp.extract_json()
        except Exception:
            # Retry once with explicit formatting fix instruction
            retry_prompt = (
                user_prompt +
                "\n\nIMPORTANT: Your previous output was not valid JSON. "
                "Respond with ONLY a valid, parseable JSON object matching the requested schema."
            )
            try:
                resp = self.llm.generate(prompt=retry_prompt, system_instruction=REVIEW_SYSTEM_PROMPT)
                return resp.extract_json()
            except Exception:
                return None
