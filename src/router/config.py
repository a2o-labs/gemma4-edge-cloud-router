from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgeLLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDGE_LLM_", env_file=".env", extra="ignore")

    base_url: str = "http://localhost:18080/v1"
    api_key: str = "edge-unused"
    model: str = "gemma4-4b"
    request_timeout_s: float = 15.0


class CloudLLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLOUD_LLM_", env_file=".env", extra="ignore")

    base_url: str = "http://127.0.0.1:4000/v1"
    api_key: str = "replace-with-litellm-key"
    model: str = "claude-opus-4-7"
    request_timeout_s: float = 60.0


class ClassifierSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLASSIFIER_", env_file=".env", extra="ignore")

    confidence_threshold: float = 0.6
    fallback: str = "heavy"


class RouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTER_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    mask_map_ttl_seconds: int = Field(default=3600, alias="MASK_MAP_TTL_SECONDS")


class Settings:
    def __init__(self) -> None:
        self.router = RouterSettings()
        self.edge = EdgeLLMSettings()
        self.cloud = CloudLLMSettings()
        self.classifier = ClassifierSettings()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
