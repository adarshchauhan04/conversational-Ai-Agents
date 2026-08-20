import pytest
from app.agent.tools import calculate, web_search, get_current_datetime
from app.agent.core import agent_instance


def test_calculator_tool_basic():
    """Test basic arithmetic calculations."""
    res = calculate.invoke("25 * 4 + 10")
    assert "Result: 110" in res


def test_calculator_tool_percentage():
    """Test percentage calculation."""
    res = calculate.invoke("15% of 200")
    assert "Result: 30" in res


def test_calculator_tool_math_functions():
    """Test math functions like sqrt."""
    res = calculate.invoke("sqrt(144)")
    assert "Result: 12" in res


def test_datetime_tool():
    """Test datetime retrieval and offset calculation."""
    res_now = get_current_datetime.invoke("now")
    assert "Current Local Date & Time" in res_now or "ISO Format" in res_now

    res_offset = get_current_datetime.invoke("45 days from today")
    assert "Calculated Date" in res_offset


def test_web_search_tool():
    """Test web search tool execution."""
    res = web_search.invoke("Python programming language")
    assert len(res) > 0
    assert "Python" in res


def test_agent_execution():
    """Test agent execution and structured response returning."""
    response = agent_instance.run(
        message="What is 50 * 3 plus 15% tip?",
        session_id="test-session-1"
    )
    assert response.session_id == "test-session-1"
    assert response.response is not None
    assert len(response.response) > 0
    assert len(response.thought_process) > 0
