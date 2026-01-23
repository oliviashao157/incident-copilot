"""Configuration management using pydantic-settings."""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o-mini"

    # GitHub Configuration
    github_token: Optional[str] = None

    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG Configuration
    rag_top_k: int = 5
    similarity_threshold: float = 0.3

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Streamlit Configuration
    streamlit_port: int = 8501

    # Paths
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_data_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def classifier_dir(self) -> Path:
        return self.artifacts_dir / "classifier"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"

    @property
    def prompts_dir(self) -> Path:
        return self.project_root / "prompts"

    def get_llm_api_key(self) -> str:
        """Get the API key for the configured LLM provider."""
        if self.llm_provider == LLMProvider.ANTHROPIC:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key
        else:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return self.openai_api_key

    def get_llm_model(self) -> str:
        """Get the model name for the configured LLM provider."""
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.llm_model
        return self.openai_model


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
