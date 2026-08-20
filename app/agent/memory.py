from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.schemas import ChatMessageSchema, ToolExecutionTrace

logger = logging.getLogger(__name__)


class SessionMemoryManager:
    """
    Manages multi-turn conversation context and history across sessions.
    Maintains session history in-memory and supports windowing to keep within context token limits.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        # Dictionary mapping session_id -> list of raw message dictionaries
        self._sessions: Dict[str, List[Dict]] = {}

    def get_or_create_session(self, session_id: str) -> List[Dict]:
        """Retrieve existing message list for a session_id or initialize a new session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        """Add a human/user message to session history."""
        session = self.get_or_create_session(session_id)
        session.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_calls": None
        })
        self._trim_session_if_needed(session_id)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        tool_calls: Optional[List[ToolExecutionTrace]] = None
    ) -> None:
        """Add an assistant message and its tool execution traces to session history."""
        session = self.get_or_create_session(session_id)
        session.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_calls": [t.model_dump() if hasattr(t, "model_dump") else t for t in (tool_calls or [])]
        })
        self._trim_session_if_needed(session_id)

    def get_langchain_messages(
        self,
        session_id: str,
        system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        """
        Convert stored session history into LangChain message objects (SystemMessage, HumanMessage, AIMessage).
        """
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        raw_msgs = self.get_or_create_session(session_id)
        for msg in raw_msgs:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        return messages

    def get_history_schemas(self, session_id: str) -> List[ChatMessageSchema]:
        """Return history as structured Pydantic ChatMessageSchema objects for API responses."""
        raw_msgs = self.get_or_create_session(session_id)
        result = []
        for msg in raw_msgs:
            traces = None
            if msg.get("tool_calls"):
                traces = [
                    t if isinstance(t, ToolExecutionTrace) else ToolExecutionTrace(**t)
                    for t in msg["tool_calls"]
                ]
            result.append(
                ChatMessageSchema(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"],
                    tool_calls=traces
                )
            )
        return result

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a specific session."""
        if session_id in self._sessions:
            self._sessions[session_id] = []
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List active session IDs."""
        return list(self._sessions.keys())

    def _trim_session_if_needed(self, session_id: str) -> None:
        """Enforce maximum conversation turns windowing to prevent context overflow."""
        session = self._sessions.get(session_id, [])
        # Each turn consists of user message + assistant response (2 messages)
        max_messages = self.max_turns * 2
        if len(session) > max_messages:
            # Keep oldest messages if system, or trim oldest user/assistant pair
            self._sessions[session_id] = session[-max_messages:]


# Global memory manager singleton instance
memory_manager = SessionMemoryManager(max_turns=10)
