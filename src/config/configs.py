"""
Main project settings.
"""

from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
import os
from pathlib import Path

import torch

class Settings(BaseSettings):
    """
    Project settings class.
    Used for centralized configuration storage.
    """
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS: int = 4

    
    # Columns for item details
    ITEM_DETAILS_COLUMNS: List[str] = [
        'category_name', 
        'im_desc', 
        'pack', 
        'bottle_volume_ml',
        'state_bottle_cost',
        'state_bottle_retail',
        'sale_dollars',
        'sale_bottles',
        'sale_liters',
    ]

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    class Config:
        env_file = ".env"


settings = Settings()

