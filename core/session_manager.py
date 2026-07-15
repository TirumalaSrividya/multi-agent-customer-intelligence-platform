"""In-memory session store.

Production note: for a real multi-process deployment behind a load
balancer, back this with Redis (session history is small, TTL-based, and
maps naturally to a Redis hash + EXPIRE) instead of an in-process dict.
The interface below is intentionally narrow so swapping the storage
backend later does not touch agent or pipeline code.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from config.settings import Settings
from core.exceptions import CapacityExceededError, SessionNotFoundError
from core.types import Role, Turn

logger = logging.getLogger("platform.session_manager")


@dataclass
class _SessionRecord:
    session_id: str
    history: list[Turn] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sessions: dict[str, _SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> _SessionRecord:
        async with self._lock:
            self._evict_expired()
            record = self._sessions.get(session_id)
            if record is None:
                if len(self._sessions) >= self.settings.max_concurrent_sessions:
                    raise CapacityExceededError(
                        f"max_concurrent_sessions={self.settings.max_concurrent_sessions} reached"
                    )
                record = _SessionRecord(session_id=session_id)
                self._sessions[session_id] = record
            record.last_active = time.time()
            return record

    async def get_history(self, session_id: str) -> list[Turn]:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            return list(record.history)

    async def append_turn(self, session_id: str, role: Role, content: str) -> None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionNotFoundError(session_id)
            record.history.append(Turn(role=role, content=content))
            max_turns = self.settings.max_history_turns
            if len(record.history) > max_turns:
                # Keep the most recent N turns. Because we always keep a
                # *contiguous suffix* of history and always send it in the
                # same order, the older portion of that suffix is still
                # byte-identical to what was sent last turn -- which is
                # exactly what lets LMCache reuse KV blocks for history
                # turn-over-turn, not just for the fixed system prompt.
                record.history = record.history[-max_turns:]
            record.last_active = time.time()

    def _evict_expired(self) -> None:
        now = time.time()
        ttl = self.settings.session_ttl_seconds
        expired = [sid for sid, rec in self._sessions.items() if now - rec.last_active > ttl]
        for sid in expired:
            logger.info("evicting expired session %s", sid)
            del self._sessions[sid]

    def active_session_count(self) -> int:
        return len(self._sessions)
