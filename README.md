# Conversational AI Agent with Hugging Face & OpenAI (2026)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hub%20Inference-yellow.svg)](https://huggingface.co/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2-green.svg)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B.svg)](https://streamlit.io/)

A production-grade Conversational AI Agent system featuring support for **Hugging Face open-access models** (`mistralai/Mistral-7B-Instruct-v0.3`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, etc.) and **OpenAI API**, combined with **custom tool-calling** (Web Search + Calculator + Datetime Engine), **multi-turn session memory**, a **FastAPI backend**, and an interactive **Streamlit UI**.

---

## 🌟 Key Capabilities

1. **Hugging Face Hub 🤗 & OpenAI ⚡ Provider Support**:
   - Easily switch between Hugging Face open-source models and OpenAI models using `.env` settings or interactive UI sidebar toggles.
   - Requires a free Hugging Face token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

2. **Tool-Calling Agent**:
   - 🔍 **Web Search Tool**: Live web search with DuckDuckGo API.
   - 🧮 **Calculator Tool**: Solves math expressions, tip/tax percentages (`18% tip on $140`), trigonometry, and roots.
   - 📅 **Datetime Tool**: Calculates date offsets (`45 days from today`) and timezone/clock info.

3. **Multi-Turn Conversation Memory**:
   - Retains turn history and conversation context per `session_id` using `SessionMemoryManager`.

4. **FastAPI Backend & Streamlit Chat Interface**:
   - Structured JSON API endpoints (`/api/v1/chat`, `/api/v1/history/{id}`, `/health`).
   - Streamlit Chat dashboard with expandable **Thought Process & Tool Inspector Drawers**.

---

## 🚀 How to Configure Hugging Face

### 1. Set Your Hugging Face API Token in `.env`

Copy `.env.example` to `.env` and set your `HF_TOKEN` / `HUGGINGFACE_MODEL`:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=hf_your_hugging_face_user_access_token
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

*(You can also use models like `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`)*.

---

## 🖥️ Running the Application

### Start Streamlit Chat Interface (UI)
```bash
python main.py --mode ui
```
Open **`http://localhost:8501`** in your browser. Select **Hugging Face Hub 🤗** in the sidebar.

### Start FastAPI REST API Server
```bash
python main.py --mode api
```
Access Swagger UI at **`http://localhost:8000/docs`**.

### Run Pytest Test Suite
```bash
python main.py --mode test
```
