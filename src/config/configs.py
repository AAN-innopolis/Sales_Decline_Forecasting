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

    ITEM_DETAILS_COLUMNS: List[str] = [
        'category_name', 
        'im_desc', 
        'pack', 
        'bottle_volume_ml','state_bottle_cost','state_bottle_retail',
        'sale_dollars','sale_bottles','sale_liters',
    ]


    class Config:
        env_file = ".env"


settings = Settings()

