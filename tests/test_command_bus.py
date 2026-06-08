"""阶段45: Agent命令总线系统 - 单元测试"""
import pytest
from src.agent.command_bus import (
    CommandStatus, Command, CommandResult, MiddlewareResult,
    CommandBus, create_command_bus,
)

class TestCommandBus:
    def test_register_and_dispatch(self):
        bus = CommandBus()
        bus.register("echo", lambda cmd: CommandResult(success=True, result=cmd.payload.get("msg")))
        r = bus.dispatch_simple("echo", {"msg": "hello"})
        assert r.success and r.result == "hello"

    def test_unknown_command(self):
        bus = CommandBus()
        r = bus.dispatch_simple("unknown")
        assert not r.success

    def test_handler_error(self):
        bus = CommandBus()
        bus.register("fail", lambda cmd: (_ for _ in ()).throw(ValueError("boom")))
        r = bus.dispatch_simple("fail")
        assert not r.success

    def test_middleware_short_circuit(self):
        bus = CommandBus()
        bus.add_middleware(lambda cmd: MiddlewareResult(result=CommandResult(success=True, result="blocked")))
        bus.register("echo", lambda cmd: CommandResult(success=True, result="should not reach"))
        r = bus.dispatch_simple("echo")
        assert r.result == "blocked"

    def test_middleware_pass_through(self):
        bus = CommandBus()
        bus.add_middleware(lambda cmd: MiddlewareResult(result=None))
        bus.register("echo", lambda cmd: CommandResult(success=True, result="ok"))
        r = bus.dispatch_simple("echo")
        assert r.result == "ok"

    def test_history(self):
        bus = CommandBus()
        bus.register("ping", lambda cmd: CommandResult(success=True))
        bus.dispatch_simple("ping")
        bus.dispatch_simple("ping")
        history = bus.get_history("ping")
        assert len(history) == 2

    def test_unregister(self):
        bus = CommandBus()
        bus.register("temp", lambda cmd: CommandResult(success=True))
        assert bus.unregister("temp")
        assert not bus.dispatch_simple("temp").success

    def test_stats(self):
        bus = CommandBus()
        bus.register("ping", lambda cmd: CommandResult(success=True))
        bus.dispatch_simple("ping")
        stats = bus.get_stats()
        assert stats["total_commands"] == 1

    def test_command_to_dict(self):
        cmd = Command(name="test", payload={"key": "val"})
        d = cmd.to_dict()
        assert d["name"] == "test"

    def test_create_command_bus(self):
        bus = create_command_bus()
        assert isinstance(bus, CommandBus)
