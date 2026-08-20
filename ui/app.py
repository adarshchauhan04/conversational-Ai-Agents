import json
import os
import sys
import time
import uuid
import requests
import streamlit as st

# Add parent directory to python path for direct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.schemas import ChatRequest, StructuredResponse
from app.agent.core import agent_instance
from app.agent.memory import memory_manager

# Page Configuration
st.set_page_config(
    page_title="Conversational AI Agent | Streamlit UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS if available
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"

if "last_structured_response" not in st.session_state:
    st.session_state.last_structured_response = None

# Sidebar Controls
st.sidebar.markdown("## 🤖 Agent Model & Provider")

# LLM Provider Selector: Hugging Face vs. OpenAI vs. Fallback
selected_provider = st.sidebar.radio(
    "Active LLM Provider",
    options=["Hugging Face Hub 🤗", "OpenAI API ⚡", "Local Reasoning Engine 🛠️"],
    index=0 if settings.has_hf_key or settings.llm_provider == "huggingface" else (1 if settings.has_openai_key else 2),
    help="Select Hugging Face Hub open-access models or OpenAI."
)

hf_token_input = ""
hf_model_input = settings.hf_model

if "Hugging Face" in selected_provider:
    st.sidebar.markdown("#### 🤗 Hugging Face Settings")
    hf_token_input = st.sidebar.text_input(
        "Hugging Face User Access Token (HF_TOKEN)",
        value=settings.effective_hf_token or "",
        type="password",
        help="Get your free access token at huggingface.co/settings/tokens"
    )
    hf_model_input = st.sidebar.text_input(
        "Hugging Face Model Repository",
        value=settings.hf_model,
        help="E.g., mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct"
    )
    if hf_token_input:
        os.environ["HF_TOKEN"] = hf_token_input
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token_input
        settings.hf_token = hf_token_input
        settings.huggingfacehub_api_token = hf_token_input
    if hf_model_input:
        settings.hf_model = hf_model_input

def is_fastapi_online(url: str) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False

api_online = is_fastapi_online(settings.fastapi_url)

execution_mode = st.sidebar.radio(
    "Connection Mode",
    options=["Direct Agent Module", "FastAPI Backend API"],
    index=1 if api_online else 0,
    help="Direct Mode executes agent logic directly inside Streamlit. FastAPI Mode connects to external FastAPI server."
)

api_base_url = st.sidebar.text_input(
    "FastAPI Server URL",
    value=settings.fastapi_url,
    help="Target host URL for FastAPI endpoint calls."
)
if not api_online and execution_mode == "FastAPI Backend API":
    st.sidebar.caption("⚠️ *FastAPI server on port 8000 is offline. Start it via `python main.py --mode api` or `python main.py --mode dev`.*")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Session & Context Controls")

active_session_id = st.sidebar.text_input(
    "Active Session ID",
    value=st.session_state.session_id,
    help="Change Session ID to test multi-turn conversation memory isolation."
)
if active_session_id != st.session_state.session_id:
    st.session_state.session_id = active_session_id

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.button("➕ New Session", use_container_width=True):
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
        st.session_state.last_structured_response = None
        st.rerun()

with col_s2:
    if st.button("🗑️ Clear History", use_container_width=True):
        memory_manager.clear_session(st.session_state.session_id)
        st.session_state.last_structured_response = None
        st.success("Session history cleared!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ LLM Hyperparameters")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.5,
    value=float(settings.temperature),
    step=0.1
)

system_prompt_override = st.sidebar.text_area(
    "System Prompt Override",
    value="",
    placeholder="Default system prompt will be used if left blank...",
    height=80
)

# System Health & Status Indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status Overview")

if "Hugging Face" in selected_provider:
    hf_status = "🟢 Token Configured" if settings.has_hf_key else "🟡 Token Required (hf.co/settings/tokens)"
    st.sidebar.info(f"**Provider:** Hugging Face Hub 🤗\n\n**Model:** `{hf_model_input}`\n\n**Status:** {hf_status}")
elif "OpenAI" in selected_provider:
    oa_status = "🟢 Key Active" if settings.has_openai_key else "🟡 API Key Required"
    st.sidebar.info(f"**Provider:** OpenAI API ⚡\n\n**Model:** `{settings.openai_model}`\n\n**Status:** {oa_status}")
else:
    st.sidebar.info("**Provider:** Local Fallback Engine 🛠️\n\n*Deterministic Tool Reasoning Mode*")

st.sidebar.caption("🔨 **Loaded Tools:** `web_search`, `calculate`, `get_current_datetime`")


# Main Page Header
st.markdown("<h1 class='main-header'>Conversational AI Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Powered by <strong>Hugging Face Hub 🤗</strong>, <strong>LangChain</strong>, <strong>FastAPI Backend</strong>, and <strong>Streamlit UI</strong>. "
    "Features tool-calling reasoning, multi-turn memory, and structured JSON output.</p>",
    unsafe_allow_html=True
)

# Quick Prompt Selector Chips
st.markdown("##### 💡 Example Queries (Click to run):")
c1, c2, c3, c4 = st.columns(4)

prompt_to_run = None

with c1:
    if st.button("🧮 Tip & Tax Calculation", use_container_width=True):
        prompt_to_run = "What is a 18% tip on $125.00 plus 8.25% tax?"

with c2:
    if st.button("🔍 Search Tech News", use_container_width=True):
        prompt_to_run = "What are the latest features in Hugging Face models and Python?"

with c3:
    if st.button("📅 Date Calculation", use_container_width=True):
        prompt_to_run = "What date will it be 45 days from today?"

with c4:
    if st.button("💬 Test Memory Context", use_container_width=True):
        prompt_to_run = "Remember that my favorite model provider is Hugging Face."


# Main Tabs Layout: Chat UI | Raw JSON Inspector | Session History
tab_chat, tab_json, tab_history = st.tabs(["💬 Agent Chat", "📦 Structured JSON Response", "📜 Session Transcript"])

with tab_chat:
    history_messages = memory_manager.get_history_schemas(st.session_state.session_id)

    chat_container = st.container()
    with chat_container:
        if not history_messages:
            st.info("👋 Hello! Ask me any question requiring Hugging Face model reasoning, live search, calculations, date queries, or multi-turn chat.")
        
        for msg in history_messages:
            if msg.role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg.content)
            elif msg.role == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg.content)
                    
                    if msg.tool_calls:
                        with st.expander("🛠️ View Invoked Tools & Traces", expanded=False):
                            for trace in msg.tool_calls:
                                st.markdown(f"**Tool:** `{trace.tool_name}` | **Time:** `{trace.execution_time_ms:.1f}ms` | **Status:** `{trace.status}`")
                                st.json({"input": trace.tool_input, "output": trace.tool_output})

    user_input = st.chat_input("Ask the agent anything...") or prompt_to_run

    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤖 Agent is processing query, executing tools, and running LLM inference..."):
                provider_key = "huggingface" if "Hugging Face" in selected_provider else ("openai" if "OpenAI" in selected_provider else "fallback")
                structured_res: StructuredResponse = None

                if execution_mode == "FastAPI Backend API":
                    try:
                        payload = {
                            "message": user_input,
                            "session_id": st.session_state.session_id,
                            "temperature": temperature,
                            "system_prompt_override": system_prompt_override if system_prompt_override.strip() else None,
                            "provider_override": provider_key,
                            "hf_model_override": hf_model_input
                        }
                        res = requests.post(
                            f"{api_base_url.rstrip('/')}/api/v1/chat",
                            json=payload,
                            timeout=5
                        )
                        if res.status_code == 200:
                            structured_res = StructuredResponse(**res.json())
                        else:
                            st.error(f"FastAPI Backend error ({res.status_code}): {res.text}")
                            structured_res = agent_instance.run(
                                message=user_input,
                                session_id=st.session_state.session_id,
                                temperature=temperature,
                                system_prompt=system_prompt_override if system_prompt_override.strip() else None,
                                provider_override=provider_key,
                                hf_model_override=hf_model_input
                            )
                    except Exception as http_err:
                        st.warning(f"Could not connect to FastAPI server at '{api_base_url}' ({http_err}). Running via Direct Module...")
                        structured_res = agent_instance.run(
                            message=user_input,
                            session_id=st.session_state.session_id,
                            temperature=temperature,
                            system_prompt=system_prompt_override if system_prompt_override.strip() else None,
                            provider_override=provider_key,
                            hf_model_override=hf_model_input
                        )
                else:
                    structured_res = agent_instance.run(
                        message=user_input,
                        session_id=st.session_state.session_id,
                        temperature=temperature,
                        system_prompt=system_prompt_override if system_prompt_override.strip() else None,
                        provider_override=provider_key,
                        hf_model_override=hf_model_input
                    )

                st.session_state.last_structured_response = structured_res
                st.markdown(structured_res.response)

                if structured_res.tool_calls or structured_res.thought_process:
                    with st.expander("🛠️ Tool Executions & Thought Process", expanded=True):
                        if structured_res.thought_process:
                            st.markdown("#### 🧠 Agent Thought Process:")
                            for step in structured_res.thought_process:
                                st.markdown(f"<div class='thought-step'>➔ {step}</div>", unsafe_allow_html=True)
                        
                        if structured_res.tool_calls:
                            st.markdown("#### 🔧 Executed Tools:")
                            for trace in structured_res.tool_calls:
                                badge_class = "tool-badge-success" if trace.status == "success" else "tool-badge"
                                st.markdown(
                                    f"<span class='tool-badge {badge_class}'>⚙️ {trace.tool_name} ({trace.execution_time_ms:.1f} ms)</span>",
                                    unsafe_allow_html=True
                                )
                                st.json({
                                    "tool_name": trace.tool_name,
                                    "input": trace.tool_input,
                                    "output": trace.tool_output,
                                    "status": trace.status
                                })

        st.rerun()

with tab_json:
    st.markdown("### 📦 Structured JSON Output Payload")
    if st.session_state.last_structured_response:
        res_dict = st.session_state.last_structured_response.model_dump()
        st.json(res_dict)
        st.download_button(
            label="📥 Download JSON Response",
            data=json.dumps(res_dict, indent=2),
            file_name=f"agent_response_{st.session_state.session_id}.json",
            mime="application/json"
        )
    else:
        st.info("No response generated yet in this session.")

with tab_history:
    st.markdown(f"### 📜 Conversation History Transcript (`{st.session_state.session_id}`)")
    full_history = memory_manager.get_history_schemas(st.session_state.session_id)
    if full_history:
        for idx, item in enumerate(full_history, 1):
            st.markdown(f"**Turn {idx} [{item.role.upper()}]** - *{item.timestamp}*")
            st.text(item.content)
            if item.tool_calls:
                st.caption(f"Tools attached: {[t.tool_name for t in item.tool_calls]}")
            st.markdown("---")
    else:
        st.info("No history recorded for active session.")
