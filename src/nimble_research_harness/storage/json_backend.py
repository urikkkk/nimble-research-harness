"""JSON file storage backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from ..models.evidence import Claim, EvidenceItem, VerificationResult
from ..models.execution import RunCheckpoint, ToolCallRecord
from ..models.output import ResearchReport, SessionSummary
from ..models.plan import ResearchPlan
from ..models.session import SessionConfig
from ..models.skill import DeploymentRecord, DynamicSkillSpec


class JsonStorageBackend:
    """File-based JSON storage in .research_sessions/."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(
            base_dir or os.environ.get("NRH_SESSIONS_DIR", ".research_sessions")
        )

    def _session_dir(self, session_id: str) -> Path:
        d = self.base_dir / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_json(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(path)

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _append_json(self, path: Path, item: dict) -> None:
        items = self._read_json(path) or []
        items.append(item)
        self._write_json(path, items)

    async def create_session(self, config: SessionConfig) -> str:
        sid = str(config.session_id)
        d = self._session_dir(sid)
        self._write_json(d / "session.json", config.model_dump(mode="json"))
        return sid

    async def load_session(self, session_id: str) -> Optional[SessionConfig]:
        data = self._read_json(self._session_dir(session_id) / "session.json")
        return SessionConfig(**data) if data else None

    async def save_skill(self, skill: DynamicSkillSpec) -> None:
        d = self._session_dir(str(skill.session_id))
        self._write_json(d / "skill.json", skill.model_dump(mode="json"))

    async def load_skill(self, session_id: str) -> Optional[DynamicSkillSpec]:
        data = self._read_json(self._session_dir(session_id) / "skill.json")
        return DynamicSkillSpec(**data) if data else None

    async def save_deployment(self, record: DeploymentRecord) -> None:
        d = self._session_dir(str(record.session_id))
        self._write_json(d / "deployment.json", record.model_dump(mode="json"))

    async def save_plan(self, plan: ResearchPlan) -> None:
        d = self._session_dir(str(plan.session_id))
        self._write_json(d / "plan.json", plan.model_dump(mode="json"))

    async def load_plan(self, session_id: str) -> Optional[ResearchPlan]:
        data = self._read_json(self._session_dir(session_id) / "plan.json")
        return ResearchPlan(**data) if data else None

    async def insert_tool_call(self, record: ToolCallRecord) -> None:
        d = self._session_dir(str(record.session_id))
        self._append_json(d / "tool_calls.json", record.model_dump(mode="json"))

    async def get_tool_calls(self, session_id: str) -> list[ToolCallRecord]:
        data = self._read_json(self._session_dir(session_id) / "tool_calls.json")
        return [ToolCallRecord(**r) for r in (data or [])]

    async def insert_evidence(self, item: EvidenceItem) -> None:
        d = self._session_dir(str(item.session_id))
        path = d / "evidence.json"
        # Deduplicate by source_url
        if item.source_url:
            existing = self._read_json(path) or []
            if any(e.get("source_url") == item.source_url for e in existing):
                return  # skip duplicate URL
        self._append_json(path, item.model_dump(mode="json"))

    async def get_evidence(self, session_id: str) -> list[EvidenceItem]:
        data = self._read_json(self._session_dir(session_id) / "evidence.json")
        return [EvidenceItem(**e) for e in (data or [])]

    async def insert_claim(self, claim: Claim) -> None:
        d = self._session_dir(str(claim.session_id))
        self._append_json(d / "claims.json", claim.model_dump(mode="json"))

    async def get_claims(self, session_id: str) -> list[Claim]:
        data = self._read_json(self._session_dir(session_id) / "claims.json")
        return [Claim(**c) for c in (data or [])]

    async def insert_verification(self, result: VerificationResult) -> None:
        d = self._session_dir(str(result.session_id))
        self._append_json(d / "verifications.json", result.model_dump(mode="json"))

    async def get_verifications(self, session_id: str) -> list[VerificationResult]:
        data = self._read_json(self._session_dir(session_id) / "verifications.json")
        return [VerificationResult(**v) for v in (data or [])]

    async def save_report(self, report: ResearchReport) -> None:
        d = self._session_dir(str(report.session_id))
        self._write_json(d / "report.json", report.model_dump(mode="json"))

    async def load_report(self, session_id: str) -> Optional[ResearchReport]:
        data = self._read_json(self._session_dir(session_id) / "report.json")
        return ResearchReport(**data) if data else None

    async def save_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        d = self._session_dir(str(checkpoint.session_id))
        cp_dir = d / "checkpoints"
        cp_dir.mkdir(exist_ok=True)
        self._write_json(
            cp_dir / f"stage_{checkpoint.stage.value}.json",
            checkpoint.model_dump(mode="json"),
        )
        self._write_json(d / "latest_checkpoint.json", checkpoint.model_dump(mode="json"))

    async def load_latest_checkpoint(self, session_id: str) -> Optional[RunCheckpoint]:
        data = self._read_json(self._session_dir(session_id) / "latest_checkpoint.json")
        return RunCheckpoint(**data) if data else None

    async def save_summary(self, summary: SessionSummary) -> None:
        d = self._session_dir(str(summary.session_id))
        self._write_json(d / "summary.json", summary.model_dump(mode="json"))

    async def list_sessions(self) -> list[dict[str, Any]]:
        if not self.base_dir.exists():
            return []
        sessions = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir():
                summary_path = d / "summary.json"
                session_path = d / "session.json"
                if session_path.exists():
                    data = self._read_json(session_path)
                    summary = self._read_json(summary_path)
                    sessions.append({
                        "session_id": d.name,
                        "user_query": data.get("user_query", ""),
                        "time_budget": data.get("time_budget", ""),
                        "created_at": data.get("created_at", ""),
                        "has_report": (d / "report.json").exists(),
                        "final_stage": summary.get("final_stage") if summary else None,
                    })
        return sessions
