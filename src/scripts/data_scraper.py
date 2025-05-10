import os
import sys
import logging
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
import time
from random import randint
from datetime import datetime

import pandas as pd
from sodapy import Socrata
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.configs import settings

BATCH_SIZE: int = 10**5
INITIAL_SLEEP_TIME: int = 10
MIN_SLEEP_TIME: int = 2
MAX_SLEEP_TIME: int = 20
MAX_RETRY_ATTEMPTS: int = 3
API_DATASET_ID: str = "m3tr-qhgy"
API_DOMAIN: str = "data.iowa.gov"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_latest_date_from_last_batch(
) -> Tuple[Optional[datetime], Optional[int]]:
    """
    Get the latest date from the last batch file.
    
    Returns:
        Optional[datetime]: The latest date from the last batch, or None if no batches exist
    """
    raw_data_path = Path("data/raw")
    batch_files = sorted(raw_data_path.glob("batch_*.csv"))
    
    if not batch_files:
        return None, None
        
    try:
        last_batch = batch_files[-1]
        df = pd.read_csv(last_batch)
        latest_date = pd.to_datetime(df['date']).max()
        logger.info(f"Found latest date in last batch: {latest_date}")
        return latest_date, int(str(last_batch).split("_")[-1].split(".")[0])
    except Exception as e:
        logger.error(f"Error reading last batch file: {e}")
        return None, None


def fetch_data_batch(
        client: Socrata, 
        offset: int, 
        latest_date: Optional[datetime] = None
    ) -> pd.DataFrame:
    """
    Fetch a batch of data from the Socrata API.

    Args:
        client: Socrata client instance
        offset: Offset for pagination
        latest_date: Optional date to filter records newer than this date

    Returns:
        DataFrame containing the fetched data
    """
    query_params = {
        "order": "date",
        "offset": offset,
        "limit": BATCH_SIZE
    }
    
    if latest_date:
        query_params["where"] = f"date > '{latest_date.strftime('%Y-%m-%d')}'"
    
    results = client.get(API_DATASET_ID, **query_params)
    return pd.DataFrame.from_records(results)


def save_data_batch(
        batch_id: int, 
        latest_date: Optional[datetime] = None, 
        last_batch: Optional[int] = None
    ) -> None:
    """
    Save a batch of data to CSV file.

    Args:
        batch_id: Batch identifier
        latest_date: Optional date to filter records newer than this date
        last_batch: Optional id of the last batch file
    """
    client = Socrata(API_DOMAIN, settings.SOCRATA_API_TOKEN)
    results_df = fetch_data_batch(client, batch_id * BATCH_SIZE, latest_date)
    
    if results_df.empty:
        logger.info("No new data to save")
        return
        
    date_range = f"{results_df['date'].min()} - {results_df['date'].max()}"
    logger.info(f'Date range: {date_range}')
    batch = batch_id if last_batch is None else last_batch + batch_id
    logger.info(f'Batch: {batch}')
    
    output_path = (
        Path("data/raw") / \
        f"batch_{batch}.csv"
    )
    results_df.loc[results_df['zipcode'] == '712-2', 'zipcode'] = 51529
    results_df['zipcode'] = results_df['zipcode'].astype(int)
    results_df.to_csv(output_path, index=False)


def attempt_data_fetch(
        batch_id: int, 
        attempt: int = 0, 
        latest_date: Optional[datetime] = None, 
        last_batch: Optional[int] = None
    ) -> bool:
    """
    Attempt to fetch and save a batch of data with retry logic.

    Args:
        batch_id: Batch identifier
        attempt: Current attempt number
        latest_date: Optional date to filter records newer than this date
        last_batch: Optional id of the last batch file
    Returns:
        bool: True if successful, False otherwise
    """
    global INITIAL_SLEEP_TIME
    
    try:
        save_data_batch(batch_id, latest_date, last_batch)
        INITIAL_SLEEP_TIME = 10
        sleep_time = randint(MIN_SLEEP_TIME, MAX_SLEEP_TIME)
        time.sleep(sleep_time)
        return True
        
    except Exception as e:
        INITIAL_SLEEP_TIME *= 1.5
        batch = batch_id if last_batch is None else last_batch + batch_id
        logger.error(f"Failed to fetch batch {batch}. Retrying in {INITIAL_SLEEP_TIME} seconds")
        logger.error(str(e))
        
        if attempt < MAX_RETRY_ATTEMPTS:
            time.sleep(INITIAL_SLEEP_TIME)
            return attempt_data_fetch(
                batch_id, 
                attempt + 1, 
                latest_date, 
                last_batch
            )
        return False


def combine_batch_files() -> None:
    """
    Combine all batch CSV files into a single parquet file.
    Also creates a zip archive of the batch files if it doesn't exist.
    """
    raw_data_path = Path("data/raw")
    batch_files = sorted(raw_data_path.glob("batch_*.csv"))
    
    if not batch_files:
        logger.info("No batch files found in data/raw")
        zip_path = raw_data_path / "batch_files.zip"
        
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(raw_data_path)
            batch_files = sorted(raw_data_path.glob("batch_*.csv"))
        else:
            logger.warning("No batch files or archive found")
            return
            
    if not (raw_data_path / "batch_files.zip").exists():
        logger.info(f"Found {len(batch_files)} batch files")
        with zipfile.ZipFile(raw_data_path / "batch_files.zip", 'w') as zipf:
            for batch_file in batch_files:
                zipf.write(batch_file, batch_file.name)
        logger.info("Batch files archived to batch_files.zip")
    
    all_batches: List[pd.DataFrame] = []
    for batch_file in tqdm(batch_files, desc="Combining batches"):
        df = pd.read_csv(batch_file)
        all_batches.append(df)
    
    combined_df = pd.concat(all_batches, ignore_index=True)
    output_path = raw_data_path / "combined_data.parquet"
    combined_df.to_parquet(output_path, index=False)
    logger.info(f"Combined data saved to {output_path}")
    logger.info(f"Total rows: {len(combined_df)}")


def main() -> None:
    """
    Main function to orchestrate the data scraping process.
    Downloads data in batches and combines them into a single file.
    """
    latest_date, last_batch = get_latest_date_from_last_batch()
    
    batch_id = 0
    while True:
        if attempt_data_fetch(
            batch_id, 
            latest_date=latest_date, 
            last_batch=last_batch
        ):
            batch_id += 1
        else:
            break

        sleep_time = randint(MIN_SLEEP_TIME, MAX_SLEEP_TIME)
        time.sleep(sleep_time)

    combine_batch_files()


if __name__ == '__main__':
    main()
