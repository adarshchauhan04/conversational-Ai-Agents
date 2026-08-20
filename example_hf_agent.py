import os
import sys

# Set standard output encoding to UTF-8 for Windows console support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure app package is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent import agent_instance

# Set Hugging Face Token programmatically or leave empty to use fallback/env
# os.environ["HF_TOKEN"] = "your_hf_token_here"

def main():
    print("Running Conversational AI Agent with Hugging Face Provider...\n")

    response = agent_instance.run(
        message="What is the latest news in AI plus calculate 25% of $400?",
        session_id="hf-session-1",
        provider_override="huggingface",
        hf_model_override="mistralai/Mistral-7B-Instruct-v0.3"
    )

    print("=" * 60)
    print("AGENT RESPONSE:")
    print("=" * 60)
    print(response.response)
    print("\n" + "=" * 60)
    print("EXECUTED TOOLS:")
    for t in response.tool_calls:
        print(f"  - [{t.tool_name}] Output: {t.tool_output} (Time: {t.execution_time_ms}ms)")
    print("=" * 60)
    print("THOUGHT PROCESS:")
    for step in response.thought_process:
        print(f"  -> {step}")
    print("=" * 60)
    print("STRUCTURED DATA JSON:")
    print(response.structured_data)


if __name__ == "__main__":
    main()
