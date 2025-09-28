from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CropVision API"
    database_url: str
    secret_key: str
    access_token_expire_minutes: int
    algorithm: str

    media_root: str
    models_path: str
    model_file: str

    admin_username: str
    admin_password: str

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


def get_settings() -> Settings:
    return Settings()
