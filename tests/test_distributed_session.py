"""阶段55: Agent分布式会话 - 单元测试"""
import time
import pytest
from src.agent.distributed_session import (
    SessionData, SessionManager, create_session_manager,
)

class TestSessionData:
    def test_expiry(self):
        s = SessionData(session_id="s1", ttl=0.01)
        time.sleep(0.02)
        assert s.is_expired

    def test_touch(self):
        s = SessionData(session_id="s1", ttl=0.1)
        time.sleep(0.05)
        s.touch()
        assert not s.is_expired

    def test_attributes(self):
        s = SessionData(session_id="s1")
        s.set_attribute("name", "Alice")
        assert s.get_attribute("name") == "Alice"
        assert s.remove_attribute("name")
        assert s.get_attribute("name") is None

class TestSessionManager:
    def test_create_and_get(self):
        m = SessionManager()
        s = m.create_session(user_id="u1")
        assert len(s.session_id) == 16
        assert m.get_session(s.session_id) is not None

    def test_expired_session(self):
        m = SessionManager(default_ttl=0.01)
        s = m.create_session()
        time.sleep(0.02)
        assert m.get_session(s.session_id) is None

    def test_destroy(self):
        m = SessionManager()
        s = m.create_session()
        assert m.destroy_session(s.session_id)
        assert m.get_session(s.session_id) is None

    def test_attributes(self):
        m = SessionManager()
        s = m.create_session()
        m.set_attribute(s.session_id, "key", "value")
        assert m.get_attribute(s.session_id, "key") == "value"

    def test_extend(self):
        m = SessionManager(default_ttl=0.1)
        s = m.create_session()
        assert m.extend_session(s.session_id, 60)
        time.sleep(0.05)
        assert m.get_session(s.session_id) is not None

    def test_list_sessions(self):
        m = SessionManager()
        s1 = m.create_session(user_id="u1")
        s2 = m.create_session(user_id="u1")
        m.create_session(user_id="u2")
        assert len(m.list_sessions("u1")) == 2

    def test_stats(self):
        m = SessionManager(default_ttl=0.01)
        m.create_session()
        time.sleep(0.02)
        stats = m.get_stats()
        assert stats["active_sessions"] == 0
        assert stats["cleaned_expired"] == 1

    def test_clear_all(self):
        m = SessionManager()
        m.create_session()
        m.create_session()
        assert m.clear_all() == 2

    def test_create_session_manager(self):
        m = create_session_manager()
        assert isinstance(m, SessionManager)
