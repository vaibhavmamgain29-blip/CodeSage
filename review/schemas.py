from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FindingCategory(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    TEST = "test"


class FindingSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Finding(BaseModel):
    file: str = Field(description="Relative path to the affected file")
    line: int = Field(description="Specific line number where the issue occurs")
    category: FindingCategory = Field(description="Category of the finding")
    severity: FindingSeverity = Field(description="Severity level: low, medium, or high")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    issue: str = Field(description="Clear explanation of the problem")
    suggestion: str = Field(description="Actionable fix or improvement")


class ReviewOutput(BaseModel):
    summary: str = Field(description="High-level overview of the pull request changes and review")
    findings: List[Finding] = Field(default_factory=list, description="List of findings discovered in the diff")
