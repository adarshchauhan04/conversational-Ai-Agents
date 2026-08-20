import pytest
from app.agent.memory import SessionMemoryManager


def test_session_memory_add_and_retrieve():
    """Test adding messages and retrieving session history."""
    mem = SessionMemoryManager(max_turns=5)
    sid = "test-mem-session"

    mem.add_user_message(sid, "Hello, agent!")
    mem.add_assistant_message(sid, "Hello, human! How can I help?")

    history = mem.get_history_schemas(sid)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hello, agent!"
    assert history[1].role == "assistant"
    assert history[1].content == "Hello, human! How can I help?"


def test_session_memory_clearing():
    """Test session clearing."""
    mem = SessionMemoryManager(max_turns=5)
    sid = "test-clear-session"

    mem.add_user_message(sid, "Message to clear")
    assert len(mem.get_history_schemas(sid)) == 1

    cleared = mem.clear_session(sid)
    assert cleared is True
    assert len(mem.get_history_schemas(sid)) == 0


def test_session_memory_trimming():
    """Test windowed memory trimming when exceeding max_turns."""
    mem = SessionMemoryManager(max_turns=2) # 2 turns = 4 messages max
    sid = "test-trim-session"

    for i in range(5):
        mem.add_user_message(sid, f"User msg {i}")
        mem.add_assistant_message(sid, f"Assistant msg {i}")

    history = mem.get_history_schemas(sid)
    assert len(history) <= 4
    assert history[-1].content == "Assistant msg 4"
