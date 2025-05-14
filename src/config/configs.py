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
    SOCRATA_API_TOKEN: str = ''
    API_KEY: str = ''
    BASE_URL: str = ''

    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS: int = 4

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    ### BUSINESS RULES
    #### TRANSACTIONS DATASET ####
    PRIMARY_KEYS: List[str] = [
        'invoice_line_no',
    ]
    NOTNA_COLUMNS: List[str] = [
        'invoice_line_no',
        'store',
        'name',
        'date', 
        'pack',
        'bottle_volume_ml',
        'state_bottle_cost',
        'sale_bottles',
        'sale_dollars',
        'sale_liters'
    ]
    NONZERO_COLUMNS: List[str] = [
        'sale_bottles',
        'sale_bottles',
        'sale_liters',
        'bottle_volume_ml',
        'state_bottle_cost',
        'pack',
    ]
    INDEXED_COLUMNS: List[str] = [
        'category', 
        'itemno', 
        'zipcode',
    ]
    CATEGORICAL_COLUMNS: List[str] = [
        'category_name',
        'name',
        'address',
        'city',
        'county', 
        'im_desc',
        'store_location',
    ]
    ### LLM FEATURE
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
    ##### AGGREGATED DATASET #####
    STATIC_CATEGORIES: List[str] = [
        "name", 
        "address", 
        "city", 
        "zipcode", 
        "county",
    ]
    STATIC_NUM: List[str] = [
        "lon", "lat",
    ]

    ### LAGS AND ROLLING FEATURES
    LAG_PERIODS: List[int] = [
        7, 14, 21
    ]
    ROLLING_WINDOW_SIZES: List[str] = [
        '7D', '14D', '21D', '30D', '60D', '90D'
    ]

    class Config:
        env_file = ".env"


settings = Settings()

