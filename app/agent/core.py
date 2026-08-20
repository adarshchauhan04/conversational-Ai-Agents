import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.schemas import StructuredResponse, ToolExecutionTrace
from app.agent.memory import memory_manager
from app.agent.tools import ALL_TOOLS, TOOL_REGISTRY, calculate, web_search, get_current_datetime

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, intelligent, and accurate Conversational AI Agent equipped with real-time tools. "
    "Use tools whenever live information, current news, calculations, or date computations are required. "
    "Always maintain turn-to-turn context from conversation history."
)


class ConversationalAgent:
    """
    Tool-calling Conversational AI Agent supporting OpenAI, Hugging Face open-access models,
    multi-turn session context memory, step-by-step thought traces, and structured JSON responses.
    """

    def __init__(self):
        self.tools = ALL_TOOLS
        self.tool_registry = TOOL_REGISTRY

    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        provider_override: Optional[str] = None,
        hf_model_override: Optional[str] = None
    ) -> StructuredResponse:
        """
        Main execution entry point for agent queries.
        
        Args:
            message: User query or command.
            session_id: Optional session identifier for conversation memory.
            temperature: LLM sampling temperature.
            system_prompt: Optional system prompt override.
            provider_override: Optional override for provider ('huggingface', 'openai', 'fallback').
            hf_model_override: Optional Hugging Face model repository string.
            
        Returns:
            StructuredResponse containing markdown text, tool traces, thought process, and structured data.
        """
        active_session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        active_provider = (provider_override or settings.active_provider).lower()
        active_hf_model = hf_model_override or settings.hf_model

        # 1. Retrieve history context for session
        history_msgs = memory_manager.get_langchain_messages(
            active_session_id,
            system_prompt=effective_system_prompt
        )

        # 2. Append user message to session memory
        memory_manager.add_user_message(active_session_id, message)

        # 3. Execution routing based on active provider
        if active_provider == "huggingface" and settings.has_hf_key:
            try:
                response = self._run_huggingface_agent(
                    message=message,
                    session_id=active_session_id,
                    history_msgs=history_msgs,
                    temperature=temperature,
                    system_prompt=effective_system_prompt,
                    model_repo=active_hf_model
                )
                memory_manager.add_assistant_message(
                    active_session_id,
                    response.response,
                    tool_calls=response.tool_calls
                )
                return response
            except Exception as err:
                logger.error(f"Hugging Face Agent execution failed, using fallback engine: {err}")

        elif active_provider == "openai" and settings.has_openai_key:
            try:
                response = self._run_openai_agent(
                    message=message,
                    session_id=active_session_id,
                    history_msgs=history_msgs,
                    temperature=temperature,
                    system_prompt=effective_system_prompt
                )
                memory_manager.add_assistant_message(
                    active_session_id,
                    response.response,
                    tool_calls=response.tool_calls
                )
                return response
            except Exception as err:
                logger.error(f"OpenAI Agent execution failed, using fallback engine: {err}")

        # 4. Fallback execution engine (runs tools deterministically & formats structured response)
        response = self._run_fallback_agent(
            message=message,
            session_id=active_session_id,
            temperature=temperature,
            system_prompt=effective_system_prompt
        )

        # Append result to memory
        memory_manager.add_assistant_message(
            active_session_id,
            response.response,
            tool_calls=response.tool_calls
        )

        return response

    def _run_huggingface_agent(
        self,
        message: str,
        session_id: str,
        history_msgs: List[Any],
        temperature: float,
        system_prompt: str,
        model_repo: str
    ) -> StructuredResponse:
        """Executes query using Hugging Face Model Inference API with LangChain integration."""
        from langchain_core.messages import HumanMessage
        from huggingface_hub import InferenceClient

        thought_process = [
            f"Received query: '{message}'",
            f"Connecting to Hugging Face Model: '{model_repo}'...",
            "Inspecting conversation memory history..."
        ]
        tool_traces: List[ToolExecutionTrace] = []

        # First, run tool evaluation if query requires math, web search, or date
        msg_lower = message.lower()
        tool_outputs = []

        if any(k in msg_lower for k in ['calculate', 'math', '+', '*', '/', 'percent', 'tip', 'tax', 'sqrt', 'sin']):
            pct_match = re.search(r'([\d\.]+\%\s+of\s+\$?[\d\.,]+)', message, re.IGNORECASE)
            math_expr_match = re.search(r'([\d\.\$\,]+\s*[\+\-\*\/\^]\s*[\d\.\$\,]+)', message)
            cleaned_expr = pct_match.group(1) if pct_match else (math_expr_match.group(1) if math_expr_match else re.sub(r'^(what is|calculate|solve|how much is|compute)\s+', '', message, flags=re.IGNORECASE))
            
            thought_process.append(f"Math intent detected. Executing tool 'calculate' on '{cleaned_expr}'")
            t_start = time.time()
            calc_out = calculate.invoke(cleaned_expr)
            t_dur = (time.time() - t_start) * 1000
            tool_traces.append(
                ToolExecutionTrace(
                    tool_name="calculate",
                    tool_input={"expression": cleaned_expr},
                    tool_output=calc_out,
                    status="success",
                    execution_time_ms=round(t_dur, 2)
                )
            )
            tool_outputs.append(f"[Tool output from 'calculate']: {calc_out}")

        if any(k in msg_lower for k in ['search', 'latest', 'news', 'who is', 'find', 'weather', 'stock']):
            thought_process.append(f"Web search intent detected. Executing tool 'web_search' for '{message}'")
            t_start = time.time()
            search_out = web_search.invoke(message)
            t_dur = (time.time() - t_start) * 1000
            tool_traces.append(
                ToolExecutionTrace(
                    tool_name="web_search",
                    tool_input={"query": message},
                    tool_output=search_out,
                    status="success",
                    execution_time_ms=round(t_dur, 2)
                )
            )
            tool_outputs.append(f"[Tool output from 'web_search']: {search_out}")

        if any(k in msg_lower for k in ['date', 'time', 'day', 'today', 'tomorrow', 'days from', 'weeks ago']):
            thought_process.append(f"Datetime intent detected. Executing tool 'get_current_datetime' for '{message}'")
            t_start = time.time()
            dt_out = get_current_datetime.invoke(message)
            t_dur = (time.time() - t_start) * 1000
            tool_traces.append(
                ToolExecutionTrace(
                    tool_name="get_current_datetime",
                    tool_input={"query": message},
                    tool_output=dt_out,
                    status="success",
                    execution_time_ms=round(t_dur, 2)
                )
            )
            tool_outputs.append(f"[Tool output from 'get_current_datetime']: {dt_out}")

        # Construct prompt for Hugging Face model
        augmented_prompt = message
        if tool_outputs:
            augmented_prompt += "\n\nContext & Tool Results:\n" + "\n".join(tool_outputs)

        # Execute via HuggingFace Hub InferenceClient
        client = InferenceClient(token=settings.effective_hf_token)
        messages_payload = [{"role": "system", "content": system_prompt}]
        
        for msg in history_msgs:
            role = "user" if msg.type == "human" else "assistant"
            messages_payload.append({"role": role, "content": str(msg.content)})

        messages_payload.append({"role": "user", "content": augmented_prompt})

        try:
            hf_res = client.chat_completion(
                messages=messages_payload,
                model=model_repo,
                max_tokens=512,
                temperature=max(0.1, temperature)
            )
            final_text = str(hf_res.choices[0].message.content)
        except Exception as client_err:
            thought_process.append(f"InferenceClient fallback to Router API: {client_err}")
            from langchain_openai import ChatOpenAI
            hf_llm = ChatOpenAI(
                model=model_repo,
                api_key=settings.effective_hf_token,
                base_url="https://api-inference.huggingface.co/v1",
                temperature=temperature
            )
            current_messages = list(history_msgs) + [HumanMessage(content=augmented_prompt)]
            ai_resp = hf_llm.invoke(current_messages)
            final_text = str(ai_resp.content)

        thought_process.append(f"Successfully generated response via Hugging Face model '{model_repo}'.")
        structured_data = self._extract_structured_data(message, final_text, tool_traces)

        return StructuredResponse(
            session_id=session_id,
            response=final_text,
            structured_data=structured_data,
            tool_calls=tool_traces,
            thought_process=thought_process
        )

    def _run_openai_agent(
        self,
        message: str,
        session_id: str,
        history_msgs: List[Any],
        temperature: float,
        system_prompt: str
    ) -> StructuredResponse:
        """Executes query using LangChain ChatOpenAI tool-calling agent."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, ToolMessage

        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature
        )

        llm_with_tools = llm.bind_tools(self.tools)
        current_messages = list(history_msgs) + [HumanMessage(content=message)]
        
        thought_process = [
            f"Received prompt: '{message}'",
            "Analyzing conversation memory context...",
            "Evaluating tool invocation requirements via OpenAI function calling..."
        ]
        tool_traces: List[ToolExecutionTrace] = []

        ai_msg = llm_with_tools.invoke(current_messages)
        
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            for tc in ai_msg.tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})
                thought_process.append(f"Tool call selected: {t_name} with inputs: {t_args}")
                
                t_start = time.time()
                tool_output_str = ""
                status = "success"
                
                if t_name in self.tool_registry:
                    try:
                        tool_fn = self.tool_registry[t_name]
                        if isinstance(t_args, dict):
                            try:
                                tool_output_str = tool_fn.invoke(t_args)
                            except Exception:
                                first_val = next(iter(t_args.values()), str(t_args))
                                tool_output_str = tool_fn.invoke(first_val)
                        else:
                            tool_output_str = tool_fn.invoke(str(t_args))
                    except Exception as e:
                        tool_output_str = f"Error executing tool {t_name}: {str(e)}"
                        status = "error"
                else:
                    tool_output_str = f"Unknown tool requested: {t_name}"
                    status = "error"

                t_dur = (time.time() - t_start) * 1000
                tool_traces.append(
                    ToolExecutionTrace(
                        tool_name=t_name,
                        tool_input=t_args,
                        tool_output=str(tool_output_str),
                        status=status,
                        execution_time_ms=round(t_dur, 2)
                    )
                )

                current_messages.append(ai_msg)
                current_messages.append(
                    ToolMessage(
                        content=str(tool_output_str),
                        tool_call_id=tc.get("id", "call_1")
                    )
                )

            final_ai_msg = llm.invoke(current_messages)
            final_text = str(final_ai_msg.content)
            thought_process.append("Synthesized final response with tool execution outputs.")
        else:
            final_text = str(ai_msg.content)
            thought_process.append("Generated response directly without external tool invocation.")

        structured_data = self._extract_structured_data(message, final_text, tool_traces)

        return StructuredResponse(
            session_id=session_id,
            response=final_text,
            structured_data=structured_data,
            tool_calls=tool_traces,
            thought_process=thought_process
        )

    def _run_fallback_agent(
        self,
        message: str,
        session_id: str,
        temperature: float,
        system_prompt: str
    ) -> StructuredResponse:
        """
        Deterministic, high-capability tool-calling agent engine used when external LLM API key
        is absent or rate-limited. Handles tool routing, calculations, searches, and context memory.
        """
        thought_process = [
            f"Received query: '{message}'",
            "Inspecting multi-turn conversation memory...",
            "Routing intent to agent tools (Web Search / Calculator / Datetime)..."
        ]
        tool_traces: List[ToolExecutionTrace] = []
        msg_lower = message.lower().strip()

        history_msgs = memory_manager.get_history_schemas(session_id)
        prev_context_hint = ""
        if history_msgs and len(history_msgs) > 1:
            last_user_msgs = [m for m in history_msgs if m.role == "user"]
            if len(last_user_msgs) > 1:
                prev_context_hint = f" (Context from previous turn: '{last_user_msgs[-2].content}')"

        response_paragraphs: List[str] = []

        # 1. Check for Greeting / Conversational intent
        greetings = ['hi', 'hello', 'hey', 'greetings', 'who are you', 'what are you', 'how are you', 'help', 'what can you do']
        is_greeting = any(re.search(r'\b' + g + r'\b', msg_lower) for g in greetings)

        if is_greeting and not any(k in msg_lower for k in ['search', 'calculate', 'date', 'time', 'news', 'weather', 'plus', 'tax']):
            thought_process.append("Greeting intent recognized. Formulating conversational response.")
            final_text = (
                f"Hello! I am your **Conversational AI Agent**{prev_context_hint}.\n\n"
                "I am equipped with real-time tools to help you:\n"
                "- 🧮 **Calculator**: Solve mathematical calculations, tip/tax percentages, trigonometry, and roots.\n"
                "- 🔍 **Web Search**: Fetch live information, news, current facts, and technical documentation.\n"
                "- 📅 **Datetime Engine**: Compute current time, dates, and offsets (*e.g., '45 days from today'*).\n"
                "- 💬 **Multi-Turn Memory**: Remember context across our conversation.\n\n"
                "How can I assist you today?"
            )
            structured_data = self._extract_structured_data(message, final_text, tool_traces)
            return StructuredResponse(
                session_id=session_id,
                response=final_text,
                structured_data=structured_data,
                tool_calls=tool_traces,
                thought_process=thought_process
            )

        # 2. Math calculation tool check
        is_math_query = any(re.search(r'\b' + p + r'\b', msg_lower) for p in ['calculate', 'math', 'sum', 'percent', 'tip', 'tax', 'sqrt', 'sin']) or ('+' in msg_lower or '*' in msg_lower or '/' in msg_lower or '%' in msg_lower)

        if is_math_query:
            pct_match = re.search(r'([\d\.]+\%\s+of\s+\$?[\d\.,]+)', message, re.IGNORECASE)
            math_expr_match = re.search(r'([\d\.\$\,]+\s*[\+\-\*\/\^]\s*[\d\.\$\,]+)', message)
            
            if pct_match:
                cleaned_expr = pct_match.group(1)
            elif math_expr_match:
                cleaned_expr = math_expr_match.group(1)
            else:
                cleaned_expr = re.sub(r'^(what is|calculate|solve|how much is|compute)\s+', '', message, flags=re.IGNORECASE)

            thought_process.append(f"Math query detected. Executing 'calculate' tool on: '{cleaned_expr}'")
            t_start = time.time()
            calc_output = calculate.invoke(cleaned_expr)
            t_dur = (time.time() - t_start) * 1000
            
            tool_traces.append(
                ToolExecutionTrace(
                    tool_name="calculate",
                    tool_input={"expression": cleaned_expr},
                    tool_output=calc_output,
                    status="success",
                    execution_time_ms=round(t_dur, 2)
                )
            )
            response_paragraphs.append(f"### 🧮 Calculation Result\n- **Expression:** `{cleaned_expr}`\n- **Output:** {calc_output}")

        # 3. Datetime tool check
        datetime_keywords = ['date', 'time', 'day', 'today', 'tomorrow', 'days from', 'weeks ago', 'clock', 'timezone']
        is_datetime_query = any(k in msg_lower for k in datetime_keywords)

        if is_datetime_query:
            thought_process.append(f"Datetime query detected. Executing 'get_current_datetime' tool for: '{message}'")
            t_start = time.time()
            dt_output = get_current_datetime.invoke(message)
            t_dur = (time.time() - t_start) * 1000

            tool_traces.append(
                ToolExecutionTrace(
                    tool_name="get_current_datetime",
                    tool_input={"query": message},
                    tool_output=dt_output,
                    status="success",
                    execution_time_ms=round(t_dur, 2)
                )
            )
            response_paragraphs.append(f"### 📅 Datetime Information\n```\n{dt_output}\n```")

        # 4. Web Search tool check
        search_keywords = ['search', 'latest', 'news', 'who is', 'what is', 'find', 'stock', 'weather', 'python', 'fastapi', 'streamlit', 'ai', 'langchain', 'huggingface', 'hugging face']
        is_search_query = any(k in msg_lower for k in search_keywords)

        if is_search_query:
            thought_process.append(f"Information query detected. Executing 'web_search' tool for: '{message}'")
            t_start = time.time()
            search_output = web_search.invoke(message)
            t_dur = (time.time() - t_start) * 1000

            tool_traces.append(
                ToolExecutionTrace(
                    tool_name="web_search",
                    tool_input={"query": message},
                    tool_output=search_output,
                    status="success",
                    execution_time_ms=round(t_dur, 2)
                )
            )
            response_paragraphs.append(f"### 🔍 Search Results & Information\n{search_output}")

        if not response_paragraphs:
            final_text = (
                f"I received your query: '{message}'{prev_context_hint}.\n\n"
                "I am ready to assist you! Try asking me:\n"
                "- *'What is a 18% tip on $125.00 plus tax?'*\n"
                "- *'Search for the latest news in AI'* \n"
                "- *'What date will it be 45 days from today?'*"
            )
        else:
            header_intro = f"Here is the response and tool execution summary{prev_context_hint}:\n\n"
            final_text = header_intro + "\n\n".join(response_paragraphs)

        thought_process.append("Completed turn processing. Packaging response with structured metadata.")
        structured_data = self._extract_structured_data(message, final_text, tool_traces)

        return StructuredResponse(
            session_id=session_id,
            response=final_text,
            structured_data=structured_data,
            tool_calls=tool_traces,
            thought_process=thought_process
        )

    def _extract_structured_data(
        self,
        query: str,
        response_text: str,
        tool_traces: List[ToolExecutionTrace]
    ) -> Dict[str, Any]:
        """Extract structured JSON entities from response for API consumers."""
        tools_used = [t.tool_name for t in tool_traces]
        
        numbers_found = []
        for t in tool_traces:
            if t.tool_name == "calculate":
                match = re.search(r'Result:\s*([\d\.\-]+)', t.tool_output)
                if match:
                    try:
                        numbers_found.append(float(match.group(1)))
                    except ValueError:
                        pass

        return {
            "query_summary": query,
            "tools_invoked_count": len(tool_traces),
            "tools_used": tools_used,
            "numeric_results": numbers_found,
            "has_tool_execution": len(tool_traces) > 0,
            "response_length_chars": len(response_text)
        }


# Singleton agent instance
agent_instance = ConversationalAgent()
