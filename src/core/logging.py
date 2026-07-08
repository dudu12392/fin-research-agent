"""Structured logging via structlog — JSON output with timestamp/level/module/message."""

from __future__ import annotations

import logging

import structlog

from core.config import settings


def setup_logging() -> None:
    """Configure structlog with JSON output including timestamp, level, module, message."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


# ═══════════════════════════════════════════════════════════════════
# Audit logging for RAG pipeline
# ═══════════════════════════════════════════════════════════════════

import os
import uuid
from datetime import datetime
from pathlib import Path

_log: Any = None
_session_log_path: Path | None = None


def _get_audit():
    global _log
    if _log is None:
        _log = __import__("structlog").get_logger("audit")
    return _log


def _write_to_session(event_name: str, **kwargs) -> None:
    """Write audit event JSON to session log file."""
    global _session_log_path
    if _session_log_path is None:
        return
    import json as _json
    from datetime import datetime as _dt
    entry = {"timestamp": _dt.now().isoformat(), "event": event_name, **kwargs}
    with open(_session_log_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")


def init_session() -> str:
    """Create session directory and log file under logs/."""
    global _session_log_path
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    log_dir = Path("logs") / session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    _session_log_path = log_dir / "audit.jsonl"
    return session_id


def log_retrieval(query: str, results: list[dict], latency_ms: float, session_id: str = "") -> None:
    """Log retrieval step."""
    companies = list({r["metadata"].get("company", "?") for r in results})
    _get_audit().info(
        "retrieval",
        session=session_id,
        query=query[:100],
        result_count=len(results),
        companies=companies,
        latency_ms=round(latency_ms, 1),
    )
    _write_to_session("retrieval", query=query[:100], result_count=len(results),
                      companies=companies, latency_ms=round(latency_ms, 1), session=session_id)


def log_llm_call(prompt: str, response: str, model: str, latency_ms: float, token_count: int = 0, session_id: str = "") -> None:
    """Log LLM generation step."""
    _get_audit().info(
        "llm_call",
        session=session_id,
        model=model,
        response_len=len(response),
        token_estimate=token_count or len(response) // 4,
        latency_ms=round(latency_ms, 1),
    )
    _write_to_session("llm_call", model=model, response_len=len(response),
                      token_estimate=token_count or len(response) // 4,
                      latency_ms=round(latency_ms, 1), session=session_id)


def log_tool_call(tool_name: str, params: dict, result: dict, session_id: str = "") -> None:
    """Log tool invocation."""
    _get_audit().info(
        "tool_call",
        session=session_id,
        tool=tool_name,
        params=str(params)[:200],
        status=result.get("status", "?"),
    )
