"""WSA catalog cache — loads and indexes available Nimble WSAs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from ..infra.logging import get_logger
from ..models.discovery import WSACandidate
from ..nimble.provider import NimbleProvider

logger = get_logger(__name__)

DEFAULT_CACHE_TTL = int(os.environ.get("NRH_WSA_CACHE_TTL", "3600"))
CACHE_DIR = Path(os.environ.get("NRH_WSA_CACHE_DIR", ".wsa_cache"))


def infer_wsa_input_params(entity_type: str | None, domain: str | None = None) -> dict[str, str]:
    """Infer the expected input params for a WSA based on its entity_type.

    Returns a dict of {param_name: description} that the planner can use
    to construct valid WSA calls without fetching each agent's schema at runtime.
    """
    et = (entity_type or "").lower()

    # SERP / Search / Listing agents → keyword-based input
    if any(k in et for k in ["serp", "search", "listing", "dealer", "directory"]):
        params = {"keyword": "Search term / product name / query string"}
        # Some agents also accept location
        if any(k in et for k in ["location", "store", "dealer"]):
            params["zipcode"] = "ZIP code for location-based results (optional)"
        elif domain and any(d in (domain or "").lower() for d in ["walmart", "target", "kroger", "heb"]):
            params["zipcode"] = "ZIP code for store location (optional)"
        return params

    # PDP / Detail / Property agents → URL-based input
    if any(k in et for k in ["detail", "pdp", "product", "property", "article", "profile", "review", "event"]):
        return {"url": "Full URL of the product/detail page to extract"}

    # Category / Landing pages
    if any(k in et for k in ["category", "landing", "clp"]):
        return {"url": "Category page URL to extract listings from"}

    # Store locator
    if any(k in et for k in ["locator", "location"]):
        return {
            "location": "City, state, or address to search near",
            "zipcode": "ZIP code (alternative to location)",
        }

    # Default fallback for unknown types
    return {"keyword": "Search term or query"}


class WSACatalog:
    """Manages the cached inventory of available Nimble WSAs."""

    def __init__(self, provider: NimbleProvider, cache_ttl: int = DEFAULT_CACHE_TTL):
        self.provider = provider
        self.cache_ttl = cache_ttl
        self._agents: list[WSACandidate] = []
        self._by_domain: dict[str, list[WSACandidate]] = {}
        self._by_vertical: dict[str, list[WSACandidate]] = {}
        self._loaded = False

    async def load(self, force_refresh: bool = False) -> None:
        if self._loaded and not force_refresh:
            return

        cached = self._load_from_disk()
        if cached and not force_refresh:
            self._agents = cached
            self._index()
            self._loaded = True
            logger.info("wsa_catalog_loaded_from_cache", count=len(self._agents))
            return

        try:
            all_agents = []
            offset = 0
            while True:
                batch = await self.provider.list_agents(limit=100, offset=offset)
                if not batch:
                    break
                for a in batch:
                    all_agents.append(
                        WSACandidate(
                            name=a.name,
                            display_name=a.display_name,
                            description=a.description,
                            vertical=a.vertical,
                            entity_type=a.entity_type,
                            domain=a.domain,
                            managed_by=a.managed_by,
                        )
                    )
                if len(batch) < 100:
                    break
                offset += 100

            self._agents = all_agents
            self._index()
            self._save_to_disk()
            self._loaded = True
            logger.info("wsa_catalog_loaded_from_api", count=len(self._agents))
        except Exception as e:
            logger.warning("wsa_catalog_load_failed", error=str(e))
            if cached:
                self._agents = cached
                self._index()
                self._loaded = True

    def _index(self) -> None:
        self._by_domain = {}
        self._by_vertical = {}
        for a in self._agents:
            if a.domain:
                self._by_domain.setdefault(a.domain.lower(), []).append(a)
            if a.vertical:
                self._by_vertical.setdefault(a.vertical.lower(), []).append(a)

    def search_by_domain(self, domain: str) -> list[WSACandidate]:
        domain = domain.lower().replace("www.", "")
        results = []
        for key, agents in self._by_domain.items():
            if domain in key or key in domain:
                results.extend(agents)
        return results

    def search_by_vertical(self, vertical: str) -> list[WSACandidate]:
        return self._by_vertical.get(vertical.lower(), [])

    def search_by_keyword(self, keyword: str) -> list[WSACandidate]:
        kw = keyword.lower()
        return [
            a
            for a in self._agents
            if kw in a.name.lower()
            or kw in (a.description or "").lower()
            or kw in (a.display_name or "").lower()
        ]

    @property
    def all_agents(self) -> list[WSACandidate]:
        return self._agents

    @property
    def count(self) -> int:
        return len(self._agents)

    def _cache_path(self) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / "wsa_catalog.json"

    def _save_to_disk(self) -> None:
        data = {
            "timestamp": time.time(),
            "agents": [a.model_dump(mode="json") for a in self._agents],
        }
        self._cache_path().write_text(json.dumps(data, indent=2))

    def _load_from_disk(self) -> Optional[list[WSACandidate]]:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            ts = data.get("timestamp", 0)
            if time.time() - ts > self.cache_ttl:
                return None
            return [WSACandidate(**a) for a in data.get("agents", [])]
        except Exception:
            return None
