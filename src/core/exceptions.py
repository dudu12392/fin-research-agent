"""Custom exception classes for the research agent."""

from __future__ import annotations


class FinResearchError(Exception):
    """Base exception for all research-agent errors."""


class SECDataNotFoundError(FinResearchError):
    """Raised when requested SEC filing data cannot be found."""


class SECRateLimitError(FinResearchError):
    """Raised when SEC EDGAR rate limits are hit."""


class IntentNotRecognizedError(FinResearchError):
    """Raised when user intent cannot be classified."""


class ValidationError(FinResearchError):
    """Raised when analysis results fail validation."""


class PlannerError(FinResearchError):
    """Raised when the planner agent fails to decompose a task."""


class FetcherError(FinResearchError):
    """Raised when the data-fetching step fails."""


class ExtractionError(FinResearchError):
    """Raised when data extraction from SEC filings fails."""


class CompanyNotFoundError(FinResearchError):
    """Raised when a company ticker cannot be found on EDGAR."""


class FilingNotFoundError(FinResearchError):
    """Raised when a specific filing cannot be found for a company."""
