"""
Script for preparing data for sales decline forecasting.
Creates new features (feature engineering) and saves the resulting dataset.
"""

import argparse
import pandas as pd
import numpy as np
import holidays
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import setup_logger, load_data
from src.config.configs import settings



def aggregate_by_store_date(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Aggregates data by store and date, creating one unique set of records for each (store, date) pair.
    
    Args:
        df (pandas.DataFrame): Source dataframe with transaction-level data
        
    Returns:
        pandas.DataFrame: Aggregated dataframe with unique (store, date) pairs
    """
    logger.info("Performing data aggregation by store and date")
    logger.info("Getting static store attributes")
    # Static features for each store
    store_attributes = df.groupby('store').agg({
        'name': lambda x: x.mode()[0],
        'address': lambda x: x.mode()[0],
        'city': lambda x: x.mode()[0],
        'zipcode': lambda x: x.mode()[0],
        'store_location': lambda x: x.mode()[0],
        'county': lambda x: x.mode()[0]
    }).reset_index()
    store_locations = store_attributes['store_location'].apply(eval)
    store_attributes['lon'] = store_locations.apply(lambda x: x['coordinates'][0])
    store_attributes['lat'] = store_locations.apply(lambda x: x['coordinates'][1])
    logger.info(f"Obtained attributes for {len(store_attributes)} unique stores")

    logger.info("Getting item details")
    # Create list of dictionaries with item and category info for each store and date
    # (features for LLM)
    item_details = df.groupby(['store', 'date']).apply(
        lambda x: x[settings.ITEM_DETAILS_COLUMNS]
        .drop_duplicates()
        .to_dict('records'), include_groups=False
    ).reset_index(name='item_details')
    logger.info(f"Obtained item details for {len(item_details)} unique (store, date) pairs")
   
    statistics = df.groupby(['store', 'date']).agg({
        # Sales - basic metrics and distributions
        'sale_bottles': ['sum', 'mean', 'median', 'min', 'max'],
        'sale_dollars': ['sum', 'mean', 'median', 'min', 'max'],
        'sale_liters': ['sum', 'mean', 'median', 'min', 'max'],
        # Number of transactions and unique receipts
        'invoice_line_no': ['count'],
        # Statistics on bottle costs
        'state_bottle_cost': ['mean', 'median', 'min', 'max'],
        'state_bottle_retail': ['mean', 'median', 'min', 'max'],
        # Statistics on categories and items
        'category': ['nunique'],
        'itemno': ['nunique'],
        # Statistics on packs and volumes
        'pack': ['mean', 'median', 'min', 'max', 'sum'],
        'bottle_volume_ml': ['mean', 'median', 'min', 'max', 'sum']
    }).reset_index()
    statistics.columns = ['_'.join(col).strip('_') for col in statistics.columns.values]
    statistics = statistics.rename(columns={
        'invoice_line_no_count': 'transaction_count',
        'category_nunique': 'unique_categories',
        'itemno_nunique': 'unique_items',
        'sale_dollars_sum': 'sale_dollars',
        'sale_bottles_sum': 'sale_bottles',
        'sale_liters_sum': 'sale_liters',
    })
    ## Key metrics for analyzing store purchases from vendor by store and purchase date
    # Average purchase price per bottle - helps analyze purchase efficiency and track price trends
    statistics['avg_price_per_bottle'] = statistics['sale_dollars'] / statistics['sale_bottles'].replace(0, np.nan)
    # Average purchase price per liter - enables comparison of purchase prices across different volume products
    statistics['avg_price_per_liter'] = statistics['sale_dollars'] / statistics['sale_liters'].replace(0, np.nan)
    # Average number of unique items per purchase - indicates assortment diversity
    statistics['avg_items_per_transaction'] = statistics['unique_items'] / statistics['transaction_count'].replace(0, np.nan)
    # Average purchase value - important indicator of purchase volume
    statistics['avg_transaction_value'] = statistics['sale_dollars'] / statistics['transaction_count'].replace(0, np.nan)
    # Purchase margin - shows difference between recommended retail price and purchase price
    statistics['profit_margin'] = (statistics['state_bottle_retail_mean'] - statistics['state_bottle_cost_mean']) / statistics['state_bottle_retail_mean'].replace(0, np.nan)
    # Markup factor - helps analyze purchase conditions and track changes in pricing policy
    statistics['discount_factor'] = statistics['state_bottle_retail_mean'] / statistics['state_bottle_cost_mean'].replace(0, np.nan)

    df_final = pd.merge(statistics, store_attributes, on=['store'], how='left')
    df_final = pd.merge(df_final, item_details, on=['store', 'date'], how='left')
    
    logger.info(f"Data aggregated shape {df_final.shape} with unique (store, date) pairs")
    return df_final


def create_temporal_features(df, logger, country_code='US', holiday_diff_threshold=0):
    """
    Create temporal features from date features.
    
    Args:
        df (pandas.DataFrame): Source dataframe
        country_code (str): Country code for holidays (ISO 3166-1 alpha-2)
        
    Returns:
        pandas.DataFrame: Dataframe with added basic features
    """
    result_df = df.copy()
    result_df['date'] = pd.to_datetime(result_df['date'])
    logger.info("Creating temporal features")

    result_df['days_since_prev_purchase'] = result_df.groupby('store')['date'].diff().dt.days

    result_df = result_df.set_index('date')
    # Day of week (0-6)
    result_df['day_of_week'] = result_df.index.dayofweek  
    result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
    # Day of month (1-28|29|30|31)
    result_df['day_of_month'] = result_df.index.day  
    result_df['day_of_month_sin'] = np.sin(2 * np.pi * result_df['day_of_month'] / 31)
    result_df['day_of_month_cos'] = np.cos(2 * np.pi * result_df['day_of_month'] / 31)
    result_df = result_df.drop(columns=['day_of_month'])
    # Month (1-12)
    result_df['month'] = result_df.index.month  
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
    result_df = result_df.drop(columns=['month'])
    # Quarter (1-4)
    result_df['quarter'] = result_df.index.quarter  
    result_df['quarter_sin'] = np.sin(2 * np.pi * result_df['quarter'] / 4)
    result_df['quarter_cos'] = np.cos(2 * np.pi * result_df['quarter'] / 4)
    result_df = result_df.drop(columns=['quarter'])
    # Week of year (1-52)
    result_df['week_of_year'] = result_df.index.isocalendar().week  
    result_df['week_of_year_sin'] = np.sin(2 * np.pi * result_df['week_of_year'] / 52)
    result_df['week_of_year_cos'] = np.cos(2 * np.pi * result_df['week_of_year'] / 52)
    result_df = result_df.drop(columns=['week_of_year'])
    # Year | Weekend (1) or not (0)
    result_df['year'] = result_df.index.year
    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
    result_df = result_df.drop(columns=['day_of_week'])
   
    logger.info(f"Adding holiday features for country {country_code}")
    unique_years = result_df['year'].unique()
    logger.info(f"Unique years in dataset: {unique_years}")
    try:
        all_holidays = pd.concat([
            pd.DataFrame({
                'date': list(holidays.country_holidays(country_code, years=year).keys()),
                'name': list(holidays.country_holidays(country_code, years=year).values())
            }) for year in unique_years
        ])
        all_holidays['date'] = pd.to_datetime(all_holidays['date'])
        dates = result_df.index.unique()
        holiday_features = pd.DataFrame(index=dates)
        # Calculate time differences between all dates and holidays
        time_diffs = (all_holidays['date'].values[:, None] - dates.values).astype('timedelta64[D]').astype(int)
        nearest_idx = np.argmin(np.abs(time_diffs), axis=0)
        days_diff = time_diffs[nearest_idx, np.arange(len(dates))]
        mask = np.abs(days_diff) <= holiday_diff_threshold
        holiday_features['is_holiday'] = mask.astype(int)
        holiday_features['holiday_name'] = np.where(mask, all_holidays.iloc[nearest_idx]['name'].values, '')
        holiday_features['days_to_nearest_holiday'] = days_diff
        
        result_df = result_df.join(holiday_features)
        logger.info(f"Successfully added holidays for country {country_code}")
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to load holidays for country {country_code}: {e}")
        logger.warning(f"Shape of result_df: {result_df.shape}, Shape days_diff: {days_diff.shape}")
        logger.warning("Using empty holiday list")
    return result_df.reset_index()


def create_rolling_features(df, logger, target='sale_dollars'):
    """
    Create rolling features for time series with grouping by stores.
    
    Args:
        df (pandas.DataFrame): Dataframe with data
        target (str): Target variable for prediction
        
    Returns:
        pandas.DataFrame: Dataframe with added rolling features
    """
    logger.info(f"Creating rolling features for {target}")
    
    df_features = df.copy()    
    # Define lag periods and window sizes for records
    lag_periods = [1, 2, 3, 4]
    window_sizes = [2, 4, 8, 12, 30, 60, 90]
    
    for period in lag_periods:
        df_features[f'prev_{period}_purchase_{target}'] = df_features.groupby('store')[target].shift(period)
    
    for window in window_sizes:
        rolling_stats = df_features.groupby('store')[target].rolling(
            window=window, 
            min_periods=1,
            closed='left'
        ).agg(['mean', 'std', 'max', 'min', 'median'])
        rolling_stats.columns = [
            f'hist_{stat}_{window}_purchases_{target}' 
            for stat in ['mean', 'std', 'max', 'min', 'median']
        ]
        df_features = df_features.join(rolling_stats.reset_index(drop=True))
        # Calculate momentum as difference between current purchase and rolling mean of previous 'window' purchases
        df_features[f'purchase_momentum_{window}'] = (
            df_features[target] - df_features[f'hist_mean_{window}_purchases_{target}']
        )
        # Calculate percentage momentum: shows relative deviation from historical average
        # Formula: ((current/mean) - 1) * 100
        # -1 centers around 0: if current = mean, result is 0%
        df_features[f'purchase_momentum_pct_{window}'] = (
            (df_features[target] / df_features[f'hist_mean_{window}_purchases_{target}'] - 1) * 100
        )
        # Calculate average days between purchases using historical data
        df_features[f'hist_avg_days_between_purchases_{window}'] = (
            df_features.groupby('store')['days_since_prev_purchase']
            .rolling(window=window, min_periods=1, closed='left')
            .mean()
            .reset_index(drop=True)
        )
    return df_features


def create_store_features(df, logger):
    """
    Create features related to stores and locations.
    
    Args:
        df (pandas.DataFrame): Dataframe with data
        
    Returns:
        pandas.DataFrame: Dataframe with added store features
    """
    logger.info("Creating features related to stores")
    df_features = df.copy()
    store_dfs = []
    q33 = (
        df_features
        .groupby('date')['sale_dollars']
        .median()
        .expanding()
        .quantile(0.33)
        .shift(1)
    )
    q66 = (
        df_features
        .groupby('date')['sale_dollars']
        .median()
        .expanding()
        .quantile(0.66)
        .shift(1)
    )
    global_sales_quantiles = pd.concat([q33.rename('q33'), q66.rename('q66')], axis=1)
    city_means = (
        df_features
        .groupby(['city','date'])['sale_dollars'].median()
        .expanding()
        .mean()
        .shift(1)
    )
    county_means = (
        df_features
        .groupby(['county', 'date'])['sale_dollars'].median()
        .expanding()
        .mean()
        .shift(1)
    )
    for _, store in df_features.groupby('store'):
        store_df = store.set_index('date')
        
        # Calculate expanding averages with shift to avoid leakage
        store_df['store_avg_sales'] = (
            store_df['sale_dollars']
            .expanding(min_periods=1)
            .median()
            .shift(1)
        )
        store_df['store_avg_transactions'] = (
            store_df['transaction_count']
            .expanding(min_periods=1)
            .mean()
            .shift(1)
        )
        store_df['store_avg_items'] = (
            store_df['unique_items']
            .expanding(min_periods=1)
            .mean()
            .shift(1)
        )
        # Get historical quantiles for store classification
        # Get quantiles for the current store's dates
        current_quantiles = global_sales_quantiles.loc[store_df.index]
        store_df['store_size'] = 1  # Small store by default
        
        # Classify store size based on historical performance
        store_df.loc[store_df['store_avg_sales'] > current_quantiles['q66'], 'store_size'] = 3  # Large store
        store_df.loc[store_df['store_avg_sales'] > current_quantiles['q33'], 'store_size'] = 2  # Medium store
        
        # Get city and county historical averages
        current_city = store_df['city'].iloc[0]
        current_county = store_df['county'].iloc[0]
        store_df['city_avg_sales'] = city_means.loc[current_city].loc[store_df.index]
        store_df['county_avg_sales'] = county_means.loc[current_county].loc[store_df.index]
        
        # Calculate ratios using historical averages
        store_df['store_to_city_sales_ratio'] = (
            store_df['store_avg_sales'] / 
            store_df['city_avg_sales'].replace(0, np.nan)
        )
        store_df['store_to_county_sales_ratio'] = (
            store_df['store_avg_sales'] / 
            store_df['county_avg_sales'].replace(0, np.nan)
        )
        
        # Keep only new features and identifiers
        new_features = [
            'store_avg_sales', 'store_avg_transactions', 'store_avg_items',
            'store_size', 'city_avg_sales', 'county_avg_sales',
            'store_to_city_sales_ratio', 'store_to_county_sales_ratio'
        ]
        store_df = store_df[['store'] + new_features].reset_index()
        store_dfs.append(store_df)
    
    store_dfs = pd.concat(store_dfs, ignore_index=True)
    return pd.merge(df_features, store_dfs, on=['store', 'date'], how='left')


def clean_and_validate_data(df_clean, logger):
    """
    Data cleaning and validation.
    
    Args:
        df (pandas.DataFrame): Source dataframe
        
    Returns:
        pandas.DataFrame: Cleaned and validated dataframe
    """
    logger.info("Cleaning and validating data")
    
    missing_values = df_clean.isnull().sum()
    missing_columns = missing_values[missing_values > 0].index
    
    if not missing_columns.empty:
        logger.warning(f"Found missing values in {len(missing_columns.tolist())} columns: \n{missing_values[missing_values > 0].sort_values(ascending=False).iloc[:20]} over {len(df_clean)} records")
        # Fill gaps for all stores
        df_clean[missing_columns] = (df_clean
            .groupby('store')[missing_columns]
            .transform(lambda x: x.bfill()#.ffill()
                       )
        )
        missing_values = df_clean.isnull().sum()
        missing_columns = missing_values[missing_values > 0].index
        logger.info(f"Filled missing values in {len(missing_columns.tolist())} columns: \n{missing_values[missing_values > 0].sort_values(ascending=False).iloc[:20]} over {len(df_clean)} records")
    
    return df_clean


def prepare_dataset(
        df: pd.DataFrame, 
        logger):
    """
    Main data preparation function.
    
    Args:
        df (pandas.DataFrame): Source dataframe
        logger (logging.Logger): Logger instance
    """
    logger.info("Starting data preparation")
    logger.info(f'Dropped {df[df["invoice_line_no"].duplicated()].shape[0]} transactions with duplicated invoice_line_no')
    df = df.drop_duplicates(subset=['invoice_line_no'])

    df_aggregated = aggregate_by_store_date(df, logger)
    df_aggregated = create_temporal_features(df_aggregated, logger)
    df_aggregated = create_rolling_features(df_aggregated, logger)
    df_aggregated = create_store_features(df_aggregated, logger)
    
    df_aggregated = clean_and_validate_data(df_aggregated, logger)        
    logger.info(f"Final dataset shape: {df_aggregated.shape}")
    return df_aggregated



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation for sales decline forecasting')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    args = parser.parse_args()

    df = load_data('data/sazerac_df.csv')
    df = df.sort_values(by=['store', 'date'])
    logger = setup_logger(args.log_level)
    logger.info(f"Data loaded, shape: {df.shape}")
    df_aggregated = prepare_dataset(df, logger) 

    output_path = 'data/sazerac_sales_prepared.parquet'
    logger.info(f"Saving results to {output_path}")
    logger.info(f"Columns: {df_aggregated.columns}")
    df_aggregated.to_parquet(output_path)
    
    logger.info(f"Data preparation completed. Dataset saved to {output_path}")