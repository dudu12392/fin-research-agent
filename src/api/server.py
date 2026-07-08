"""FastAPI server exposing the research agent via REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.config import settings
from core.logging import setup_logging
from graph.research_graph import research_graph

setup_logging()

app = FastAPI(
    title="Fin Research Agent",
    description="AI-powered financial research agent using SEC filings",
    version="0.1.0",
)


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    intent: str
    report: str
    validation: str
    plan: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    error: str | None = None


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run a financial research query end-to-end."""
    try:
        result = await research_graph.run(request.query)
        return ResearchResponse(
            intent=result.get("intent", "unknown"),
            report=result.get("report", ""),
            validation=result.get("validation", ""),
            plan=result.get("plan"),
            data=result.get("data"),
            error=result.get("error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
