from pydantic import BaseSettings


class Settings(BaseSettings):
    app_env: str = "dev"
    log_level: str = "INFO"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_seconds: int = 3600

    admin_user: str = "admin"
    admin_password: str = "admin123"

    db_url: str = "postgresql+psycopg2://landcover:landcover@postgres:5432/landcover"

    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123"
    s3_bucket: str = "landcover"

    mlflow_tracking_uri: str = "http://mlflow:5000"

    inference_url: str = "http://inference:8001"
    training_url: str = "http://training:8002"

    redis_url: str = "redis://redis:6379/0"

    model_name: str = "snunet_cbam"
    model_stage: str = "Production"

    class Config:
        env_file = ".env"
        case_sensitive = False
