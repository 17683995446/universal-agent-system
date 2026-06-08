"""
阶段82: Agent智能对话管理系统

遵循"图难于其易，为大于其细"的原则，实现对话状态追踪、意图识别、槽位填充、上下文管理。
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class IntentType(str, Enum):
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    CONFIRM = "confirm"
    DENY = "deny"
    INFORM = "inform"
    UNKNOWN = "unknown"


class DialogueState(str, Enum):
    START = "start"
    AWAITING_INFO = "awaiting_info"
    CONFIRMING = "confirming"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Slot:
    name: str
    value: Any = None
    required: bool = True
    filled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "filled": self.filled,
        }


@dataclass
class Intent:
    intent_type: IntentType
    confidence: float
    entities: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type.value,
            "confidence": round(self.confidence, 4),
            "entities": self.entities,
        }


@dataclass
class DialogueTurn:
    turn_id: str
    user_input: str
    intent: Optional[Intent] = None
    slots: Dict[str, Slot] = field(default_factory=dict)
    response: str = ""
    state: DialogueState = DialogueState.START
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input[:50],
            "intent": self.intent.to_dict() if self.intent else None,
            "state": self.state.value,
        }


@dataclass
class DialogueSession:
    session_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    global_slots: Dict[str, Slot] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "state": self.current_state().value,
        }

    def current_state(self) -> DialogueState:
        if not self.turns:
            return DialogueState.START
        return self.turns[-1].state

    def get_slot(self, name: str) -> Optional[Any]:
        slot = self.global_slots.get(name)
        return slot.value if slot and slot.filled else None

    def set_slot(self, name: str, value: Any) -> None:
        if name not in self.global_slots:
            self.global_slots[name] = Slot(name=name)
        self.global_slots[name].value = value
        self.global_slots[name].filled = True


class IntentRecognizer:
    """意图识别器"""

    # 简单关键词规则
    INTENT_PATTERNS = {
        IntentType.GREETING: ["hello", "hi", "hey", "你好", "您好"],
        IntentType.QUESTION: ["what", "how", "why", "where", "when", "什么", "怎么", "为什么"],
        IntentType.COMMAND: ["do", "make", "create", "run", "执行", "创建", "运行"],
        IntentType.CONFIRM: ["yes", "ok", "sure", "confirm", "是", "确认", "好的"],
        IntentType.DENY: ["no", "cancel", "stop", "否", "取消", "不要"],
    }

    def recognize(self, text: str) -> Intent:
        text_lower = text.lower()
        scores = {}
        for intent_type, keywords in self.INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent_type] = score
        if not scores:
            return Intent(intent_type=IntentType.UNKNOWN, confidence=1.0)
        best = max(scores.items(), key=lambda x: x[1])
        total = sum(scores.values())
        return Intent(
            intent_type=best[0],
            confidence=best[1] / total,
            entities=self._extract_entities(text),
        )

    def _extract_entities(self, text: str) -> Dict[str, str]:
        """简单实体抽取"""
        entities = {}
        # 数字
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            entities["number"] = numbers[0]
        return entities


class SlotFiller:
    """槽位填充器"""

    def __init__(self, required_slots: List[str]):
        self._required_slots = required_slots

    def fill_slots(self, text: str, existing_slots: Dict[str, Slot]) -> Dict[str, Slot]:
        slots = {name: Slot(name=name, required=True) for name in self._required_slots}
        # 继承已有值
        for name, slot in existing_slots.items():
            if name in slots:
                slots[name] = slot

        # 简单模式匹配填充
        import re
        for name in self._required_slots:
            if not slots[name].filled:
                # 尝试从文本中提取
                pattern = rf'{name}[是为]?\s*[:：]?\s*([\w.]+)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    slots[name].value = match.group(1)
                    slots[name].filled = True
        return slots

    def get_missing_slots(self, slots: Dict[str, Slot]) -> List[str]:
        return [name for name, slot in slots.items() if slot.required and not slot.filled]


class ContextManager:
    """上下文管理器"""

    def __init__(self, max_history: int = 10):
        self._sessions: Dict[str, DialogueSession] = {}
        self._max_history = max_history

    def create_session(self) -> DialogueSession:
        session_id = str(uuid.uuid4())[:8]
        session = DialogueSession(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[DialogueSession]:
        return self._sessions.get(session_id)

    def add_turn(self, session_id: str, user_input: str) -> Optional[DialogueTurn]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        turn = DialogueTurn(
            turn_id=f"{session_id}_{len(session.turns)}",
            user_input=user_input,
        )
        session.turns.append(turn)
        session.last_active = time.time()
        # 限制历史长度
        if len(session.turns) > self._max_history:
            session.turns = session.turns[-self._max_history:]
        return turn

    def get_recent_context(self, session_id: str, n: int = 3) -> List[DialogueTurn]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.turns[-n:]

    def clear_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def get_active_sessions(self, timeout_sec: float = 3600) -> List[str]:
        now = time.time()
        return [sid for sid, s in self._sessions.items() if now - s.last_active < timeout_sec]


class DialogueManager:
    """对话管理器"""

    def __init__(self):
        self._intent_recognizer = IntentRecognizer()
        self._context_manager = ContextManager()
        self._slot_fillers: Dict[str, SlotFiller] = {}

    def register_slot_filler(self, intent_type: IntentType, filler: SlotFiller) -> None:
        self._slot_fillers[intent_type] = filler

    def start_session(self) -> str:
        session = self._context_manager.create_session()
        return session.session_id

    def process_input(self, session_id: str, user_input: str) -> DialogueTurn:
        """处理用户输入"""
        turn = self._context_manager.add_turn(session_id, user_input)
        if not turn:
            return DialogueTurn(turn_id="error", user_input=user_input, response="Session not found")

        session = self._context_manager.get_session(session_id)

        # 意图识别
        intent = self._intent_recognizer.recognize(user_input)
        turn.intent = intent

        # 槽位填充
        filler = self._slot_fillers.get(intent.intent_type)
        if filler:
            turn.slots = filler.fill_slots(user_input, session.global_slots if session else {})
            # 更新全局槽位
            if session:
                for name, slot in turn.slots.items():
                    if slot.filled:
                        session.set_slot(name, slot.value)
            missing = filler.get_missing_slots(turn.slots)
            if missing:
                turn.state = DialogueState.AWAITING_INFO
                turn.response = f"Please provide: {', '.join(missing)}"
            else:
                turn.state = DialogueState.COMPLETED
                turn.response = "All information collected."
        else:
            turn.state = DialogueState.COMPLETED
            turn.response = self._generate_response(intent)

        return turn

    def _generate_response(self, intent: Intent) -> str:
        responses = {
            IntentType.GREETING: "Hello! How can I help you?",
            IntentType.QUESTION: "I understand you have a question.",
            IntentType.COMMAND: "I'll process your command.",
            IntentType.CONFIRM: "Confirmed.",
            IntentType.DENY: "Understood. Cancelled.",
            IntentType.UNKNOWN: "I'm not sure I understand. Could you rephrase?",
        }
        return responses.get(intent.intent_type, "OK.")

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        session = self._context_manager.get_session(session_id)
        if not session:
            return {}
        return {
            "session_id": session_id,
            "turn_count": len(session.turns),
            "current_state": session.current_state().value,
            "filled_slots": {name: slot.value for name, slot in session.global_slots.items() if slot.filled},
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_sessions": len(self._context_manager._sessions),
            "registered_fillers": len(self._slot_fillers),
        }


def create_dialogue_manager() -> DialogueManager:
    return DialogueManager()
