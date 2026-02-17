from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_db: str = Field(default="prediction_agent", alias="MONGODB_DB")
    loop_interval_seconds: int = Field(default=120, alias="LOOP_INTERVAL_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    polymarket_enabled: bool = Field(default=True, alias="POLYMARKET_ENABLED")
    polymarket_gamma_base_url: str = Field(
        default="https://gamma-api.polymarket.com", alias="POLYMARKET_GAMMA_BASE_URL"
    )
    polymarket_clob_base_url: str = Field(default="https://clob.polymarket.com", alias="POLYMARKET_CLOB_BASE_URL")
    polymarket_limit: int = Field(default=200, alias="POLYMARKET_LIMIT")

    kalshi_enabled: bool = Field(default=True, alias="KALSHI_ENABLED")
    kalshi_base_url: str = Field(default="https://api.elections.kalshi.com/trade-api/v2", alias="KALSHI_BASE_URL")
    kalshi_limit: int = Field(default=200, alias="KALSHI_LIMIT")

    alpha_vantage_api_key: str = Field(default="", alias="ALPHA_VANTAGE_API_KEY")
    alpha_vantage_base_url: str = Field(default="https://www.alphavantage.co/query", alias="ALPHA_VANTAGE_BASE_URL")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    alert_top_n: int = Field(default=5, alias="ALERT_TOP_N")
    alert_min_score: float = Field(default=0.62, alias="ALERT_MIN_SCORE")
    alert_cooldown_minutes: int = Field(default=120, alias="ALERT_COOLDOWN_MINUTES")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-mini", alias="OPENAI_MODEL")
    openai_timeout_seconds: int = Field(default=20, alias="OPENAI_TIMEOUT_SECONDS")

    max_signals_per_cycle: int = Field(default=150, alias="MAX_SIGNALS_PER_CYCLE")
    strict_theme_filter: bool = Field(default=True, alias="STRICT_THEME_FILTER")
    min_signal_liquidity: float = Field(default=10000.0, alias="MIN_SIGNAL_LIQUIDITY")
    min_signal_volume_24h: float = Field(default=5000.0, alias="MIN_SIGNAL_VOLUME_24H")
    min_probability_edge: float = Field(default=0.10, alias="MIN_PROBABILITY_EDGE")

    streaming_enabled: bool = Field(default=False, alias="STREAMING_ENABLED")
    streaming_batch_seconds: int = Field(default=20, alias="STREAMING_BATCH_SECONDS")
    streaming_max_buffer: int = Field(default=1000, alias="STREAMING_MAX_BUFFER")

    polymarket_ws_enabled: bool = Field(default=True, alias="POLYMARKET_WS_ENABLED")
    polymarket_ws_url: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/market", alias="POLYMARKET_WS_URL"
    )
    polymarket_ws_market_limit: int = Field(default=100, alias="POLYMARKET_WS_MARKET_LIMIT")

    kalshi_ws_enabled: bool = Field(default=True, alias="KALSHI_WS_ENABLED")
    kalshi_ws_url: str = Field(default="wss://api.elections.kalshi.com/trade-api/ws/v2", alias="KALSHI_WS_URL")
    kalshi_ws_market_limit: int = Field(default=100, alias="KALSHI_WS_MARKET_LIMIT")
    kalshi_ws_channel: str = Field(default="ticker", alias="KALSHI_WS_CHANNEL")

    backtest_horizon_days: int = Field(default=5, alias="BACKTEST_HORIZON_DAYS")
    backtest_transaction_cost_bps: float = Field(default=10.0, alias="BACKTEST_TRANSACTION_COST_BPS")
    backtest_min_trades: int = Field(default=30, alias="BACKTEST_MIN_TRADES")
    backtest_threshold_min: float = Field(default=0.55, alias="BACKTEST_THRESHOLD_MIN")
    backtest_threshold_max: float = Field(default=0.85, alias="BACKTEST_THRESHOLD_MAX")
    backtest_threshold_step: float = Field(default=0.02, alias="BACKTEST_THRESHOLD_STEP")
    calibration_train_days: int = Field(default=120, alias="CALIBRATION_TRAIN_DAYS")
    calibration_val_days: int = Field(default=30, alias="CALIBRATION_VAL_DAYS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
