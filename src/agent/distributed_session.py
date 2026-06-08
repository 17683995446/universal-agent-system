"""
阶段55: Agent分布式会话

遵循"图难于其易，为大于其细"的原则，实现会话创建、属性管理、TTL过期、续期。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class SessionData:
    session_id: str
    user_id: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: float = 3600.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.last_accessed > self.ttl

    @property
    def remaining(self) -> float:
        return max(0, self.ttl - (time.time() - self.last_accessed))

    def touch(self) -> None:
        self.last_accessed = time.time()

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        self.touch()

    def get_attribute(self, key: str, default: Any = None) -> Any:
        self.touch()
        return self.attributes.get(key, default)

    def remove_attribute(self, key: str) -> bool:
        if key in self.attributes:
            del self.attributes[key]
            self.touch()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "ttl": self.ttl,
            "attributes": self.attributes,
        }

class SessionManager:
    """会话管理器"""
    def __init__(self, default_ttl: float = 3600.0):
        self._default_ttl = default_ttl
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.RLock()

    def create_session(self, user_id: str = "", ttl: Optional[float] = None) -> SessionData:
        with self._lock:
            session_id = str(uuid.uuid4())[:16]
            session = SessionData(
                session_id=session_id,
                user_id=user_id,
                ttl=ttl if ttl is not None else self._default_ttl,
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                return None
            session.touch()
            return session

    def destroy_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def set_attribute(self, session_id: str, key: str, value: Any) -> bool:
        with self._lock:
            session = self.get_session(session_id)
            if session:
                session.set_attribute(key, value)
                return True
            return False

    def get_attribute(self, session_id: str, key: str, default: Any = None) -> Any:
        session = self.get_session(session_id)
        return session.get_attribute(key, default) if session else default

    def extend_session(self, session_id: str, additional_ttl: float) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                session.ttl += additional_ttl
                session.touch()
                return True
            return False

    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        with self._lock:
            result = []
            for sid, session in list(self._sessions.items()):
                if session.is_expired:
                    del self._sessions[sid]
                    continue
                if user_id is None or session.user_id == user_id:
                    result.append(sid)
            return result

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            active = 0
            expired = 0
            for sid, session in list(self._sessions.items()):
                if session.is_expired:
                    expired += 1
                    del self._sessions[sid]
                else:
                    active += 1
            return {
                "active_sessions": active,
                "cleaned_expired": expired,
            }

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            return count

def create_session_manager(default_ttl: float = 3600.0) -> SessionManager:
    return SessionManager(default_ttl)
