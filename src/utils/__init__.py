import sys
from pathlib import Path
import logging

def find_project_root():
    path = Path(__file__).resolve()
    while not (path / 'pyproject.toml').exists():
        path = path.parent
        if path == path.parent:
            break
    return path


sys.path.insert(0, str(find_project_root()))

from src.utils.data_utils import (
    setup_logger, 
    load_data, 
    save_data,
    setup_seed
)


__all__ = ['setup_logger', 'load_data', 'save_data', 'setup_seed']
