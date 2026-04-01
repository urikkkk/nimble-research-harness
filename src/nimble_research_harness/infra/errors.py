"""Error hierarchy for the research harness."""


class ResearchHarnessError(Exception):
    pass


class NimbleApiError(ResearchHarnessError):
    def __init__(self, status_code: int, response_body: str, message: str = ""):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message or f"Nimble API error {status_code}: {response_body[:200]}")


class AgentTimeoutError(ResearchHarnessError):
    def __init__(self, phase_name: str, elapsed_ms: int):
        self.phase_name = phase_name
        self.elapsed_ms = elapsed_ms
        super().__init__(f"Agent timeout in phase '{phase_name}' after {elapsed_ms}ms")


class AgentAbortError(ResearchHarnessError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Agent aborted: {reason}")


class StorageError(ResearchHarnessError):
    def __init__(self, operation: str, detail: str):
        self.operation = operation
        self.detail = detail
        super().__init__(f"Storage error during '{operation}': {detail}")


class SkillGenerationError(ResearchHarnessError):
    pass


class WSAResolutionError(ResearchHarnessError):
    pass
