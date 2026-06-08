"""
阶段47: Agent状态机引擎

遵循"图难于其易，为大于其细"的原则，实现状态定义、转换规则、守卫条件、历史记录。
"""
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

class TransitionError(Exception):
    pass

@dataclass
class State:
    name: str
    on_enter: Optional[Callable[[], None]] = None
    on_exit: Optional[Callable[[], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transition:
    name: str
    from_state: str
    to_state: str
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateChangeRecord:
    from_state: str
    to_state: str
    transition_name: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

class StateMachine:
    """状态机引擎"""
    def __init__(self, initial_state: str):
        self._states: Dict[str, State] = {}
        self._transitions: Dict[str, List[Transition]] = {}
        self._current_state = initial_state
        self._context: Dict[str, Any] = {}
        self._history: List[StateChangeRecord] = []
        self._lock = threading.RLock()
        # 初始化当前状态
        self._states[initial_state] = State(name=initial_state)

    def add_state(self, state: State) -> None:
        with self._lock:
            self._states[state.name] = state

    def add_transition(self, transition: Transition) -> None:
        with self._lock:
            if transition.from_state not in self._transitions:
                self._transitions[transition.from_state] = []
            self._transitions[transition.from_state].append(transition)

    def can_transition(self, transition_name: str) -> bool:
        with self._lock:
            transitions = self._transitions.get(self._current_state, [])
            for t in transitions:
                if t.name == transition_name:
                    if t.guard and not t.guard(self._context):
                        return False
                    return True
            return False

    def transition(self, transition_name: str, context: Optional[Dict] = None) -> bool:
        with self._lock:
            if context:
                self._context.update(context)
            transitions = self._transitions.get(self._current_state, [])
            target = None
            for t in transitions:
                if t.name == transition_name:
                    if t.guard and not t.guard(self._context):
                        raise TransitionError(f"守卫条件不满足: {transition_name}")
                    target = t
                    break
            if not target:
                raise TransitionError(f"无效转换: {transition_name} (当前状态: {self._current_state})")

            old_state = self._current_state
            new_state_name = target.to_state

            # 退出当前状态
            old = self._states.get(old_state)
            if old and old.on_exit:
                try:
                    old.on_exit()
                except Exception:
                    pass

            # 执行转换动作
            if target.action:
                try:
                    target.action(self._context)
                except Exception:
                    pass

            # 进入新状态
            self._current_state = new_state_name
            new = self._states.get(new_state_name)
            if new and new.on_enter:
                try:
                    new.on_enter()
                except Exception:
                    pass

            # 记录历史
            self._history.append(StateChangeRecord(
                from_state=old_state, to_state=new_state_name,
                transition_name=transition_name, context=dict(self._context),
            ))
            return True

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def context(self) -> Dict[str, Any]:
        return dict(self._context)

    def set_context(self, key: str, value: Any) -> None:
        with self._lock:
            self._context[key] = value

    def get_available_transitions(self) -> List[str]:
        with self._lock:
            transitions = self._transitions.get(self._current_state, [])
            return [t.name for t in transitions if not t.guard or t.guard(self._context)]

    def get_history(self, limit: int = 50) -> List[StateChangeRecord]:
        return list(self._history[-limit:])

    def get_all_states(self) -> List[str]:
        return list(self._states.keys())

    def reset(self, state: str) -> None:
        with self._lock:
            self._current_state = state
            self._history.clear()

def create_state_machine(initial: str) -> StateMachine:
    return StateMachine(initial)
