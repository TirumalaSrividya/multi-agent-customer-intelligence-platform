class PlatformError(Exception):
    """Base class for all platform-raised errors."""


class LLMBackendError(PlatformError):
    """Raised when the LLM backend fails after retries (network, rate limit,
    5xx, timeout)."""


class LLMRateLimitError(LLMBackendError):
    """Raised specifically on 429 / rate-limit responses so callers can
    apply a different backoff strategy than generic failures."""


class MalformedAgentOutputError(PlatformError):
    """Raised when an agent's LLM output cannot be parsed into its expected
    schema, even after the retry-with-repair-prompt attempt."""

    def __init__(self, agent: str, raw_output: str, cause: Exception | None = None):
        self.agent = agent
        self.raw_output = raw_output
        self.cause = cause
        super().__init__(f"[{agent}] failed to parse model output: {cause}")


class KnowledgeBaseError(PlatformError):
    """Raised on vector store ingestion / query failures."""


class SessionNotFoundError(PlatformError):
    pass


class CapacityExceededError(PlatformError):
    """Raised when MAX_CONCURRENT_SESSIONS would be exceeded."""
