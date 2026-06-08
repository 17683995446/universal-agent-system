"""阶段47: Agent状态机引擎 - 单元测试"""
import pytest
from src.agent.state_machine import (
    State, Transition, StateChangeRecord, StateMachine,
    TransitionError, create_state_machine,
)

class TestStateMachine:
    def test_basic_transition(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("running"))
        sm.add_transition(Transition("start", "idle", "running"))
        sm.transition("start")
        assert sm.current_state == "running"

    def test_invalid_transition(self):
        sm = StateMachine("idle")
        sm.add_transition(Transition("start", "idle", "running"))
        with pytest.raises(TransitionError):
            sm.transition("stop")

    def test_guard_condition(self):
        sm = StateMachine("locked")
        sm.add_state(State("locked"))
        sm.add_state(State("unlocked"))
        sm.add_transition(Transition("unlock", "locked", "unlocked",
                                       guard=lambda ctx: ctx.get("key") == "valid"))
        sm.set_context("key", "invalid")
        assert not sm.can_transition("unlock")
        sm.set_context("key", "valid")
        assert sm.can_transition("unlock")
        sm.transition("unlock")
        assert sm.current_state == "unlocked"

    def test_transition_action(self):
        sm = StateMachine("idle")
        sm.add_state(State("running"))
        sm.add_transition(Transition("start", "idle", "running",
                                       action=lambda ctx: ctx.update({"started": True})))
        sm.transition("start")
        assert sm.context.get("started") is True

    def test_on_enter_exit(self):
        log = []
        sm = StateMachine("a")
        sm.add_state(State("a", on_exit=lambda: log.append("exit_a")))
        sm.add_state(State("b", on_enter=lambda: log.append("enter_b")))
        sm.add_transition(Transition("go", "a", "b"))
        sm.transition("go")
        assert log == ["exit_a", "enter_b"]

    def test_available_transitions(self):
        sm = StateMachine("idle")
        sm.add_transition(Transition("start", "idle", "running"))
        sm.add_transition(Transition("stop", "idle", "stopped"))
        avail = sm.get_available_transitions()
        assert set(avail) == {"start", "stop"}

    def test_history(self):
        sm = StateMachine("a")
        sm.add_state(State("b"))
        sm.add_state(State("c"))
        sm.add_transition(Transition("ab", "a", "b"))
        sm.add_transition(Transition("bc", "b", "c"))
        sm.transition("ab")
        sm.transition("bc")
        history = sm.get_history()
        assert len(history) == 2
        assert history[0].from_state == "a"

    def test_reset(self):
        sm = StateMachine("a")
        sm.add_state(State("b"))
        sm.add_transition(Transition("ab", "a", "b"))
        sm.transition("ab")
        sm.reset("a")
        assert sm.current_state == "a"
        assert len(sm.get_history()) == 0

    def test_create_state_machine(self):
        sm = create_state_machine("init")
        assert sm.current_state == "init"
