import time
from typing import Dict, List
from fastapi import APIRouter, HTTPException, Path, Query, status

from app.config import settings
from app.schemas import (
    ChatRequest,
    StructuredResponse,
    SessionHistoryResponse,
    ToolInfoResponse,
    ToolInfo,
    HealthResponse
)
from app.agent.core import agent_instance
from app.agent.memory import memory_manager
from app.agent.tools import ALL_TOOLS

router = APIRouter()
START_TIME = time.time()


@router.post(
    "/api/v1/chat",
    response_model=StructuredResponse,
    status_code=status.HTTP_200_OK,
    summary="Send message to Conversational AI Agent",
    description="Processes user query using multi-turn session memory and tool calling (Web Search + Calculator + Datetime), returning structured JSON."
)
async def chat_endpoint(request: ChatRequest) -> StructuredResponse:
    """
    Main Chat API Endpoint.
    Consumes a prompt and returns a structured JSON payload with markdown response, tool traces, and thought steps.
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'message' must be a non-empty string."
            )

        response = agent_instance.run(
            message=request.message,
            session_id=request.session_id,
            temperature=request.temperature or settings.temperature,
            system_prompt=request.system_prompt_override,
            provider_override=request.provider_override,
            hf_model_override=request.hf_model_override
        )
        return response

    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing error: {str(err)}"
        )


@router.get(
    "/api/v1/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Retrieve session history",
    description="Gets full conversation history and attached tool execution traces for a given session ID."
)
async def get_history_endpoint(
    session_id: str = Path(..., description="Unique session ID")
) -> SessionHistoryResponse:
    """Returns conversation transcript and turn history."""
    history = memory_manager.get_history_schemas(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=history,
        total_turns=len(history)
    )


@router.delete(
    "/api/v1/history/{session_id}",
    summary="Clear session history",
    description="Resets context memory for a given session ID."
)
async def clear_history_endpoint(
    session_id: str = Path(..., description="Unique session ID to clear")
) -> Dict[str, str]:
    """Clears history for session."""
    success = memory_manager.clear_session(session_id)
    if not success:
        return {"session_id": session_id, "message": "Session ID not found or already empty."}
    return {"session_id": session_id, "message": f"Successfully cleared history for session '{session_id}'."}


@router.get(
    "/api/v1/tools",
    response_model=ToolInfoResponse,
    summary="List available agent tools",
    description="Returns metadata, name, and description of all registered tools."
)
async def list_tools_endpoint() -> ToolInfoResponse:
    """Lists registered agent tools."""
    tool_list = []
    for t in ALL_TOOLS:
        schema_dict = t.args_schema.model_json_schema() if t.args_schema else {"description": "string query"}
        tool_list.append(
            ToolInfo(
                name=t.name,
                description=t.description,
                parameters=schema_dict
            )
        )
    return ToolInfoResponse(tools=tool_list, count=len(tool_list))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Health Status",
    description="Returns service health status, active LLM provider (OpenAI / Hugging Face / Fallback), and tool availability."
)
async def health_endpoint() -> HealthResponse:
    """Service health check."""
    active_prov = settings.active_provider
    if active_prov == "huggingface":
        llm_provider = f"Hugging Face Hub ({settings.hf_model})"
    elif active_prov == "openai":
        llm_provider = f"OpenAI API ({settings.openai_model})"
    else:
        llm_provider = "Intelligent Fallback Engine"

    tools_available = [t.name for t in ALL_TOOLS]
    
    return HealthResponse(
        status="ok",
        llm_provider=llm_provider,
        tools_available=tools_available,
        mode="active"
    )
