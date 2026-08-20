from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for conversational AI endpoint."""
    message: str = Field(
        ...,
        description="User message or prompt for the conversational agent.",
        examples=["What is the stock price of Apple today plus 15% calculation?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Unique session identifier for multi-turn history. Generated automatically if omitted.",
        examples=["session-abc-123"]
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM text generation."
    )
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Optional custom system prompt instruction."
    )


class ToolExecutionTrace(BaseModel):
    """Detailed record of a single tool execution during reasoning."""
    tool_name: str = Field(..., description="Name of the invoked tool.")
    tool_input: Any = Field(..., description="Input parameters passed to the tool.")
    tool_output: str = Field(..., description="String result returned by the tool.")
    status: str = Field(default="success", description="Execution status: 'success' or 'error'.")
    execution_time_ms: float = Field(default=0.0, description="Tool execution duration in milliseconds.")


class StructuredResponse(BaseModel):
    """
    Structured JSON response schema returned by FastAPI endpoint.
    Can be programmatically parsed by API consumers.
    """
    session_id: str = Field(..., description="Session identifier for this conversation turn.")
    response: str = Field(..., description="Formatted markdown text response generated for the user.")
    structured_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted key information, entities, or calculation summaries."
    )
    tool_calls: List[ToolExecutionTrace] = Field(
        default_factory=list,
        description="Traces of tools invoked by the agent to satisfy the query."
    )
    thought_process: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning steps recorded during agent planning."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of response generation."
    )
    tokens_used: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        description="Token consumption metrics for this turn."
    )


class ChatMessageSchema(BaseModel):
    """Representation of a stored message in conversation history."""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'.")
    content: str = Field(..., description="Text content of the message.")
    timestamp: str = Field(..., description="Timestamp when message was added.")
    tool_calls: Optional[List[ToolExecutionTrace]] = Field(default=None, description="Tool traces attached to turn.")


class SessionHistoryResponse(BaseModel):
    """Response containing full context history for a session."""
    session_id: str
    messages: List[ChatMessageSchema]
    total_turns: int


class ToolInfo(BaseModel):
    """Information metadata for an agent tool."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolInfoResponse(BaseModel):
    """Response containing list of available agent tools."""
    tools: List[ToolInfo]
    count: int


class HealthResponse(BaseModel):
    """Health check endpoint status response."""
    status: str = "ok"
    llm_provider: str
    tools_available: List[str]
    mode: str = "active"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
