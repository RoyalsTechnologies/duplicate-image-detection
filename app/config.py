from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_name: str = "did-backend-api"
    environment: str = "local"
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    allowed_ips: str = ""
    trusted_proxy_ips: str = ""
    local_storage_dir: Path = Path("uploads")
    public_base_url: str = "http://localhost:8000"
    s3_bucket: str | None = None
    aws_region: str | None = None
    api_v1_prefix: str = "/api/v1"
    port: int = 8000
    object_storage_bucket: str | None = None
    object_storage_region: str | None = None
    object_storage_access_key: str | None = None
    object_storage_secret_key: str | None = None
    cv_provider: str = "local"
    cv_api_key: str | None = None
    cv_device: str = "cpu"
    cv_embedding_model: str = "ViT-B-32"
    cv_embedding_pretrained: str = "openai"
    cv_cache_dir: Path = Path("/var/cache/did-backend-api")
    cv_yolo_model: str = "/var/cache/did-backend-api/yolo11n.pt"
    cv_yolo_confidence: float = 0.25
    cv_reject_irrelevant_images: bool = True
    cv_relevance_threshold: float = 0.15
    cv_relevance_min_detection_confidence: float = 0.25
    duplicate_distance_meters: int = 100
    duplicate_time_window_hours: int = 720
    duplicate_similarity_threshold: float = 0.86
    possible_duplicate_similarity_threshold: float = 0.68
    perceptual_hash_high_similarity: float = 0.92
    perceptual_hash_possible_similarity: float = 0.75
    max_upload_size_mb: int = 10
    upload_rate_limit_per_minute: int = 20

    @property
    def duplicate_exact_distance_meters(self) -> int:
        return min(50, self.duplicate_distance_meters)

    @property
    def duplicate_possible_distance_meters(self) -> int:
        return self.duplicate_distance_meters

    @property
    def duplicate_high_confidence(self) -> float:
        return self.duplicate_similarity_threshold

    @property
    def duplicate_possible_confidence(self) -> float:
        return self.possible_duplicate_similarity_threshold

    @property
    def allowed_ip_ranges(self) -> list[str]:
        return [item.strip() for item in self.allowed_ips.split(",") if item.strip()]

    @property
    def trusted_proxy_ranges(self) -> list[str]:
        return [item.strip() for item in self.trusted_proxy_ips.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
