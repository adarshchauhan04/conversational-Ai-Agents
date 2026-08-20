from app.agent.tools.calculator import calculate
from app.agent.tools.web_search import web_search
from app.agent.tools.datetime_tool import get_current_datetime

ALL_TOOLS = [web_search, calculate, get_current_datetime]

TOOL_REGISTRY = {
    "web_search": web_search,
    "calculate": calculate,
    "get_current_datetime": get_current_datetime
}
