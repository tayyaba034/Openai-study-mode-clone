"""
Configuration module for the Study Mode application.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google AI Studio API Configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    
    # Xero API Configuration (optional)
    xero_client_id: Optional[str] = Field(default=None, env="XERO_CLIENT_ID")
    xero_client_secret: Optional[str] = Field(default=None, env="XERO_CLIENT_SECRET")
    
    # Alternative Xero variable names (from your .env file)
    xero_client_key: Optional[str] = Field(default=None, env="XERO_CLIENT_KEY")
    xero_secret_key: Optional[str] = Field(default=None, env="XERO_SECRET_KEY")
    
    # Application Configuration
    app_host: str = Field(default="localhost", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # AI Model Configuration
    default_model: str = Field(default="gemini-pro", env="DEFAULT_MODEL")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields in .env file
    }

# Global settings instance
settings = Settings()

def get_google_api_key() -> str:
    """Get the Google AI Studio API key."""
    return settings.google_api_key

def get_xero_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get Xero API credentials."""
    # Try standard names first, then fallback to alternative names
    client_id = settings.xero_client_id or settings.xero_client_key
    client_secret = settings.xero_client_secret or settings.xero_secret_key
    return client_id, client_secret

def get_app_config() -> dict:
    """Get application configuration."""
    return {
        "host": settings.app_host,
        "port": settings.app_port,
        "debug": settings.debug
    }
