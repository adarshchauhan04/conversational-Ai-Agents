import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings supporting both OpenAI and Hugging Face providers.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Provider Selection ('openai', 'huggingface', or 'auto')
    llm_provider: str = Field(default="huggingface", validation_alias="LLM_PROVIDER")

    # Hugging Face Settings
    hf_token: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")
    huggingfacehub_api_token: Optional[str] = Field(default=None, validation_alias="HUGGINGFACEHUB_API_TOKEN")
    hf_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3", validation_alias="HUGGINGFACE_MODEL")

    # OpenAI Settings
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    temperature: float = Field(default=0.7, validation_alias="TEMPERATURE")

    # Memory Settings
    max_memory_turns: int = Field(default=10, validation_alias="MAX_MEMORY_TURNS")

    # FastAPI Server Settings
    fastapi_host: str = Field(default="0.0.0.0", validation_alias="FASTAPI_HOST")
    fastapi_port: int = Field(default=8000, validation_alias="FASTAPI_PORT")

    # Streamlit Settings
    streamlit_port: int = Field(default=8501, validation_alias="STREAMLIT_SERVER_PORT")
    fastapi_url: str = Field(default="http://localhost:8000", validation_alias="FASTAPI_URL")

    @property
    def effective_hf_token(self) -> Optional[str]:
        """Return resolved Hugging Face token."""
        token = self.hf_token or self.huggingfacehub_api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if token and token.strip() and not token.startswith("your_"):
            return token.strip()
        return None

    @property
    def has_hf_key(self) -> bool:
        """Return True if a valid Hugging Face API key/token is set."""
        return self.effective_hf_token is not None

    @property
    def has_openai_key(self) -> bool:
        """Return True if a valid OpenAI API key is set."""
        key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        return bool(key and key.strip() and not key.startswith("your_"))

    @property
    def active_provider(self) -> str:
        """Determine active provider based on configuration and key availability."""
        prov = (self.llm_provider or "huggingface").lower()
        if prov == "huggingface" and self.has_hf_key:
            return "huggingface"
        elif prov == "openai" and self.has_openai_key:
            return "openai"
        elif self.has_hf_key:
            return "huggingface"
        elif self.has_openai_key:
            return "openai"
        return "fallback"


settings = Settings()
