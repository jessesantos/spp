from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    database_url: str = Field(
        default="postgresql+asyncpg://spp:spp@postgres/spp"
    )
    redis_url: str = Field(default="redis://redis:6379/0")

    anthropic_api_key: str | None = None
    claude_model: str = Field(default="claude-sonnet-4-5-20250929")

    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    brapi_base_url: str = Field(default="https://brapi.dev/api")
    rss_feeds: list[str] = Field(
        default=[
            "https://www.infomoney.com.br/feed/",
            "https://www.moneytimes.com.br/feed/",
        ]
    )

    sentiment_cache_ttl_seconds: int = 86400
    price_cache_ttl_seconds: int = 300

    model_fallback_stub: bool = Field(default=True)
    models_dir: str = Field(default="/app/models")
    rate_limit_default: str = Field(default="60/minute")
    rate_limit_predict: str = Field(default="10/minute")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
