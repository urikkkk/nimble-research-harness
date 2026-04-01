"""All enum definitions for the research harness."""

from enum import Enum


class TimeBudget(str, Enum):
    QUICK_30S = "30s"
    SHORT_2M = "2m"
    MEDIUM_5M = "5m"
    STANDARD_10M = "10m"
    DEEP_30M = "30m"
    EXHAUSTIVE_1H = "1h"

    @property
    def seconds(self) -> int:
        return {
            "30s": 30,
            "2m": 120,
            "5m": 300,
            "10m": 600,
            "30m": 1800,
            "1h": 3600,
        }[self.value]

    @property
    def label(self) -> str:
        return {
            "30s": "30 seconds",
            "2m": "2 minutes",
            "5m": "5 minutes",
            "10m": "10 minutes",
            "30m": "30 minutes",
            "1h": "1 hour",
        }[self.value]


class ExecutionStage(str, Enum):
    INTAKE = "intake"
    DISCOVERY = "discovery"
    SKILL_GEN = "skill_gen"
    DEPLOYMENT = "deployment"
    PLANNING = "planning"
    RESEARCH = "research"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


class ClaimConfidence(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    WEAK_SUPPORT = "weak_support"
    UNRESOLVED = "unresolved"


class ReportFormat(str, Enum):
    BRIEF = "brief"
    FULL_REPORT = "full_report"
    JSON = "json"
    EVIDENCE_TABLE = "evidence_table"
    SOURCE_PACK = "source_pack"


class ExecutionMode(str, Enum):
    HYBRID = "hybrid"
    RAW_TOOLS = "raw_tools"
    WSA_ONLY = "wsa_only"


class ToolName(str, Enum):
    SEARCH = "nimble_search"
    EXTRACT = "nimble_extract"
    MAP = "nimble_map"
    CRAWL_RUN = "nimble_crawl_run"
    CRAWL_STATUS = "nimble_crawl_status"
    AGENTS_RUN = "nimble_agents_run"
    AGENTS_GET = "nimble_agents_get"
    AGENTS_LIST = "nimble_agents_list"
    TASK_RESULTS = "nimble_task_results"


class TaskType(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    MARKET_RESEARCH = "market_research"
    COMPETITIVE_INTEL = "competitive_intel"
    COMPANY_DEEP_DIVE = "company_deep_dive"
    TREND_ANALYSIS = "trend_analysis"
    DATA_COLLECTION = "data_collection"
    VERIFICATION = "verification"
    OPEN_EXPLORATION = "open_exploration"


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class SearchFocus(str, Enum):
    GENERAL = "general"
    NEWS = "news"
    CODING = "coding"
    ACADEMIC = "academic"
    SHOPPING = "shopping"
    SOCIAL = "social"
    GEO = "geo"
    LOCATION = "location"


class SearchDepth(str, Enum):
    LITE = "lite"
    FAST = "fast"
    DEEP = "deep"


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
