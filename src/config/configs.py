from pydantic_settings import BaseSettings
from typing import Dict, List, Optional

import torch

class Settings(BaseSettings):
    DB_NAME: str = 'retail'
    DB_USER: str = 'postgres'
    DB_PASSWORD: str = 'password'
    DB_HOST: str = 'localhost'
    DB_PORT: int = 5432

    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS: int = 4


    class Config:
        env_file = ".env"


settings = Settings()

