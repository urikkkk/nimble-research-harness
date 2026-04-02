"""Nimble API request/response types."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


_VALID_FOCUS = {"general", "news", "coding", "academic", "shopping", "social", "geo", "location"}
_VALID_DEPTH = {"lite", "fast", "deep"}


# --- Search ---

class SearchParams(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=100)
    search_depth: str = Field(default="lite")
    focus: str = Field(default="general")

    @field_validator("focus", mode="before")
    @classmethod
    def _sanitize_focus(cls, v: Any) -> str:
        if isinstance(v, str) and v.lower().strip() in _VALID_FOCUS:
            return v.lower().strip()
        return "general"

    @field_validator("search_depth", mode="before")
    @classmethod
    def _sanitize_depth(cls, v: Any) -> str:
        if isinstance(v, str) and v.lower().strip() in _VALID_DEPTH:
            return v.lower().strip()
        return "lite"
    include_answer: bool = False
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    locale: Optional[str] = None
    country: Optional[str] = None
    time_range: Optional[str] = None


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    content: Optional[str] = None
    position: int = 0


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    answer: Optional[str] = None
    request_id: Optional[str] = None


# --- Extract ---

class ExtractParams(BaseModel):
    url: str
    formats: list[str] = Field(default_factory=lambda: ["markdown"])
    render: bool = False
    country: Optional[str] = None
    locale: Optional[str] = None


class ExtractResponse(BaseModel):
    url: str = ""
    task_id: Optional[str] = None
    status: str = ""
    markdown: Optional[str] = None
    html: Optional[str] = None
    status_code: Optional[int] = None


# --- Map ---

class MapParams(BaseModel):
    url: str
    limit: int = Field(default=100, ge=1, le=100000)
    sitemap: str = Field(default="include")
    domain_filter: str = Field(default="all")


class MapLink(BaseModel):
    url: str = ""
    title: Optional[str] = None
    description: Optional[str] = None


class MapResponse(BaseModel):
    task_id: Optional[str] = None
    success: bool = True
    links: list[MapLink] = Field(default_factory=list)


# --- Crawl ---

class CrawlParams(BaseModel):
    url: str
    limit: int = Field(default=50, ge=1, le=10000)
    max_discovery_depth: int = Field(default=3, ge=1, le=10)
    sitemap: str = Field(default="include")
    name: Optional[str] = None


class CrawlRunResponse(BaseModel):
    crawl_id: str = ""
    status: str = ""


class CrawlStatusResponse(BaseModel):
    crawl_id: str = ""
    status: str = ""
    total: int = 0
    completed: int = 0
    failed: int = 0
    pending: int = 0
    tasks: list[dict[str, Any]] = Field(default_factory=list)


# --- Agents ---

class AgentSummary(BaseModel):
    name: str = ""
    display_name: Optional[str] = None
    description: Optional[str] = None
    vertical: Optional[str] = None
    entity_type: Optional[str] = None
    domain: Optional[str] = None
    managed_by: Optional[str] = None


class AgentDetails(BaseModel):
    name: str = ""
    display_name: Optional[str] = None
    description: Optional[str] = None
    vertical: Optional[str] = None
    entity_type: Optional[str] = None
    domain: Optional[str] = None
    input_properties: Any = Field(default_factory=list)  # API returns list, not dict
    output_schema: dict[str, Any] = Field(default_factory=dict)
    feature_flags: dict[str, bool] = Field(default_factory=dict)

    @property
    def input_params_summary(self) -> str:
        """Human-readable summary of input params for the planner."""
        if isinstance(self.input_properties, list):
            parts = []
            for p in self.input_properties:
                name = p.get("name", "?")
                req = "required" if p.get("required") else "optional"
                desc = p.get("description", "")[:60]
                parts.append(f'"{name}" ({req}): {desc}')
            return "; ".join(parts)
        return str(self.input_properties)


class AgentRunResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = ""
    data: Any = None


# --- Tasks ---

class TaskResults(BaseModel):
    task_id: str = ""
    status: str = ""
    data: Any = None
