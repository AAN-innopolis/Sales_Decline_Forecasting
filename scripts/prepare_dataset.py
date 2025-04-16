"""
Script for preparing data for sales decline forecasting.
Creates new features (feature engineering) and saves the resulting dataset.
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
import warnings

# Logging setup
def setup_logger(log_level=logging.INFO):
    """Setup logger with specified logging level."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    return logging.getLogger(__name__)

logger = setup_logger()


def load_data(data_path):
    """
    Load source data.
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])
    logger.info(f"Data loaded, shape: {df.shape}")
    return df

def create_basic_features(result_df, country_code='US'):
    """
    Create basic features from date and numeric values.
    
    Args:
        df (pandas.DataFrame): Source dataframe
        country_code (str): Country code for holidays (ISO 3166-1 alpha-2)
        
    Returns:
        pandas.DataFrame: Dataframe with added basic features
    """
    logger.info("Creating basic features from date and numeric values")
    # Set date as index
    result_df = result_df.set_index('date')
    
    # Time features
    logger.info("Adding time features")
    result_df['day_of_week'] = result_df.index.dayofweek  # Day of week (0-6)
    result_df['month'] = result_df.index.month  # Month (1-12)
    result_df['quarter'] = result_df.index.quarter  # Quarter (1-4)
    result_df['year'] = result_df.index.year  # Year
    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)  # Weekend (1) or not (0)
    result_df['day_of_month'] = result_df.index.day  # Day of month
    result_df['week_of_year'] = result_df.index.isocalendar().week  # Week of year
    
    # Holiday features using holidays library
    logger.info(f"Adding holiday features for country {country_code}")
    
    # Initialize holiday columns
    result_df['is_holiday'] = 0
    result_df['holiday_name'] = ''
    result_df['is_day_before_holiday'] = 0
    result_df['is_day_after_holiday'] = 0
    result_df['days_to_nearest_holiday'] = 30 
    
    # Get unique years from dataset
    unique_years = result_df.index.year.unique()
    logger.info(f"Unique years in dataset: {unique_years}")
    
    try:
        
        for year in unique_years:
            # Create holidays object for current year
            holidays_dict = holidays.country_holidays(country_code, years=year)
            holiday_df = pd.DataFrame({
                'date': list(holidays_dict.keys()),
                'name': list(holidays_dict.values())
            })
            holiday_df['date'] = pd.to_datetime(holiday_df['date'])
            
            # For each date in our index
            for idx_date in result_df.index.unique():
                # Find the nearest holiday
                time_diffs = (holiday_df['date'] - idx_date).abs()
                if not time_diffs.empty:  # Check for empty dataframe
                    nearest_idx = time_diffs.idxmin()
                    nearest_holiday = holiday_df.iloc[nearest_idx]
                    days_diff = (nearest_holiday['date'] - idx_date).days
                    
                    # If difference is not more than 3 days, consider it a holiday
                    if abs(days_diff) <= 3:
                        result_df.loc[idx_date, 'is_holiday'] = 1
                        result_df.loc[idx_date, 'holiday_name'] = nearest_holiday['name']
                    
                    # Add days to nearest holiday
                    result_df.loc[idx_date, 'days_to_nearest_holiday'] = days_diff
                    
                    # Check day before and after holiday
                    if days_diff == 1:  # If 1 day before holiday
                        result_df.loc[idx_date, 'is_day_before_holiday'] = 1
                    elif days_diff == -1:  # If 1 day after holiday
                        result_df.loc[idx_date, 'is_day_after_holiday'] = 1
        
        logger.info(f"Successfully added holidays for country {country_code}")
        
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to load holidays for country {country_code}: {e}")
        logger.warning("Using empty holiday list")
    
    # Price features and their ratios
    logger.info("Adding price features and ratios")
    
    result_df['avg_price_per_bottle'] = result_df['sale_dollars'] / result_df['sale_bottles'].replace(0, np.nan)
    result_df['avg_price_per_liter'] = result_df['sale_dollars'] / result_df['sale_liters'].replace(0, np.nan)
    result_df['profit_margin'] = (result_df['state_bottle_retail_mean'] - result_df['state_bottle_cost_mean']) / result_df['state_bottle_retail_mean'].replace(0, np.nan)
    result_df['discount_factor'] = result_df['state_bottle_retail_mean'] / result_df['state_bottle_cost_mean'].replace(0, np.nan)
    result_df['avg_transaction_value'] = result_df['sale_dollars'] / result_df['transaction_count'].replace(0, np.nan)
    
    # Fill NaN values
    result_df = result_df.fillna({
        'avg_price_per_bottle': result_df['avg_price_per_bottle'].median(),
        'avg_price_per_liter': result_df['avg_price_per_liter'].median(),
        'profit_margin': result_df['profit_margin'].median(),
        'discount_factor': result_df['discount_factor'].median(),
        'avg_transaction_value': result_df['avg_transaction_value'].median(),
        'days_to_nearest_holiday': 60
    })
    
    return result_df.reset_index()

def create_temporal_features(df, target='sale_dollars'):
    """
    Create temporal features for time series with grouping by stores.
    
    Args:
        df (pandas.DataFrame): Dataframe with data
        target (str): Target variable for analysis
        
    Returns:
        pandas.DataFrame: Dataframe with added temporal features
    """
    logger.info(f"Creating temporal features for {target}")
    
    df_features = df.copy()
    
    # Sort data by store and date
    df_features = df_features.sort_values(['store', 'date'])
    
    # For each store, create temporal features separately
    store_dfs = []
    
    for store, store_df in df_features.groupby('store'):
        # Set date as index for correct work with time windows
        store_df = store_df.set_index('date')
        
        # 1. Lag features
        # Use shift by index (dates)
        for shift_days in [1, 7, 14, 30, 60]:
            # Get number of days for shift
            store_df[f'{target}_lag_{shift_days}d'] = store_df[target].shift(periods=1, freq=f'{shift_days}D')
        
        # 2. Rolling statistics by actual dates
        date_windows = ['7D', '14D', '30D', '60D', '90D']
        
        for window in date_windows:
            # Use time window for calculating statistics
            rolling = store_df[target].rolling(window=window, min_periods=1)
            
            store_df[f'{target}_rolling_mean_{window}'] = rolling.mean()
            store_df[f'{target}_rolling_std_{window}'] = rolling.std()
            store_df[f'{target}_rolling_max_{window}'] = rolling.max()
            store_df[f'{target}_rolling_min_{window}'] = rolling.min()
            store_df[f'{target}_rolling_median_{window}'] = rolling.median()
            
            # Purchase frequency
            if len(store_df) > 1:
                avg_days_between = (store_df.index[-1] - store_df.index[0]).days / (len(store_df) - 1)
                store_df[f'avg_days_between_purchases_{window}'] = avg_days_between
            else:
                store_df[f'avg_days_between_purchases_{window}'] = np.nan
            
            # Trend and momentum indicators
            store_df[f'{target}_momentum_{window}'] = store_df[target] - store_df[f'{target}_rolling_mean_{window}']
            store_df[f'{target}_rel_momentum_{window}'] = (store_df[target] / store_df[f'{target}_rolling_mean_{window}'] - 1) * 100
        
        # 3. Time since last purchase features
        store_df['days_since_last_purchase'] = (store_df.index - store_df.index.shift(1, freq='D')).days
        
        # 4. Sales decline features considering irregular data
        # Define decline relative to previous purchase
        store_df[f'{target}_decrease_prev'] = (store_df[target] < store_df[f'{target}_lag_1d']).astype(int)
        
        # Define decline relative to averages over different periods
        store_df[f'{target}_decrease_7d_avg'] = (store_df[target] < store_df[f'{target}_rolling_mean_7D']).astype(int)
        store_df[f'{target}_decrease_30d_avg'] = (store_df[target] < store_df[f'{target}_rolling_mean_30D']).astype(int)
        
        # Significant decline (more than 10% from 30-day average)
        store_df[f'{target}_significant_decrease'] = (store_df[target] < store_df[f'{target}_rolling_mean_30D'] * 0.9).astype(int)
        
        # 5. Purchase trend features
        if len(store_df) >= 3:
            # Determine if the sequence of last 3 purchases is decreasing
            store_df['consecutive_decrease'] = ((store_df[target] < store_df[target].shift(1)) & 
                                                (store_df[target].shift(1) < store_df[target].shift(2))).astype(int)
        else:
            store_df['consecutive_decrease'] = 0
            
        # 6. Seasonal features
        store_df['month_sin'] = np.sin(2 * np.pi * store_df.index.month / 12)
        store_df['month_cos'] = np.cos(2 * np.pi * store_df.index.month / 12)
        store_df['day_of_month_sin'] = np.sin(2 * np.pi * store_df.index.day / 31)
        store_df['day_of_month_cos'] = np.cos(2 * np.pi * store_df.index.day / 31)
        
        # Save processed dataframe and return index to column
        store_df = store_df.reset_index()
        store_dfs.append(store_df)
    
    return pd.concat(store_dfs, ignore_index=True)

def create_store_features(df):
    """
    Create features related to stores and locations.
    
    Args:
        df (pandas.DataFrame): Dataframe with data
        
    Returns:
        pandas.DataFrame: Dataframe with added store features
    """
    logger.info("Creating features related to stores and locations")
    
    df_features = df.copy()
    
    # Processing store geolocation (extracting coordinates from dictionary)
    logger.info("Extracting coordinates from store geolocation data")
    
    # Extracting coordinates
    store_locations = df_features['store_location'].apply(eval)
    df_features['lon'] = store_locations.apply(lambda x: x['coordinates'][0])
    df_features['lat'] = store_locations.apply(lambda x: x['coordinates'][1])
    df_features = df_features.drop(columns=['store_location'])
    logger.info(f"Extracted coordinates for all {len(df_features)} records")
    
    # Average sales by stores
    logger.info("Adding average sales by stores")
    store_sales_mean = df_features.groupby('store')['sale_dollars'].transform('mean')
    df_features['store_avg_sales'] = store_sales_mean
    
    # Categorical features for stores based on average sales
    logger.info("Adding categorical features for stores")
    sales_quantiles = store_sales_mean.quantile([0.33, 0.66])
    df_features['store_size'] = 1  # small store by default
    df_features.loc[store_sales_mean > sales_quantiles[0.33], 'store_size'] = 2  # medium store
    df_features.loc[store_sales_mean > sales_quantiles[0.66], 'store_size'] = 3  # large store
    
    # Average sales by cities
    logger.info("Adding average sales by cities")
    city_sales_mean = df_features.groupby('city')['sale_dollars'].transform('mean')
    df_features['city_avg_sales'] = city_sales_mean
    
    # Average sales by counties
    logger.info("Adding average sales by counties")
    county_sales_mean = df_features.groupby('county')['sale_dollars'].transform('mean')
    df_features['county_avg_sales'] = county_sales_mean
    
    # Now transaction_count is already the number of transactions in aggregated data
    # Add average number of transactions by stores
    logger.info("Adding average number of transactions by stores")
    store_avg_transactions = df_features.groupby('store')['transaction_count'].transform('mean')
    df_features['store_avg_transactions'] = store_avg_transactions
    
    # Ratio of store average sales to city average sales
    logger.info("Adding ratio of store sales to city")
    df_features['store_to_city_sales_ratio'] = df_features['store_avg_sales'] / df_features['city_avg_sales'].replace(0, np.nan)
    
    # Ratio of store average sales to county average sales
    logger.info("Adding ratio of store sales to county")
    df_features['store_to_county_sales_ratio'] = df_features['store_avg_sales'] / df_features['county_avg_sales'].replace(0, np.nan)
    
    logger.info("Adding product assortment diversity features")
    df_features['store_avg_items'] = df_features.groupby('store')['unique_items'].transform('mean')
    # Ratio of unique items to number of transactions
    df_features['items_per_transaction_ratio'] = df_features['unique_items'] / df_features['transaction_count'].replace(0, np.nan)
    
    # Fill NaN values
    df_features = df_features.fillna({
        'store_to_city_sales_ratio': 1.0,
        'store_to_county_sales_ratio': 1.0,
        'items_per_transaction_ratio': df_features['items_per_transaction_ratio'].median()
    })
    
    return df_features

def clean_and_validate_data(df_clean):
    """
    Data cleaning and validation.
    
    Args:
        df (pandas.DataFrame): Source dataframe
        
    Returns:
        pandas.DataFrame: Cleaned and validated dataframe
    """
    logger.info("Cleaning and validating data")
    
    # Check for negative values in sales and replace with zeros
    # sales_columns = ['sale_bottles', 'sale_dollars', 'sale_liters', 'sale_gallons']
    # for col in sales_columns:
    #     neg_count = (df_clean[col] < 0).sum()
    #     if neg_count > 0:
    #         logger.warning(f"Found {neg_count} negative values in column {col}, replacing with zeros")
    #         df_clean[col] = df_clean[col].clip(lower=0)
    
    # Check for outliers in numeric columns
    # numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    # for col in numeric_cols:
    #     if col not in ['year', 'month', 'day_of_week', 'quarter', 'is_weekend', 'is_holiday', 'day_of_month', 'week_of_year','lon','lat']:
    #         # Calculate quantiles to identify outliers
    #         q1 = df_clean[col].quantile(0.01)
    #         q3 = df_clean[col].quantile(0.99)
    #         iqr = q3 - q1
    #         lower_bound = q1 - 1.5 * iqr
    #         upper_bound = q3 + 1.5 * iqr
            
    #         # Count outliers
    #         outliers_low = (df_clean[col] < lower_bound).sum()
    #         outliers_high = (df_clean[col] > upper_bound).sum()
            
    #         if outliers_low > 0 or outliers_high > 0:
    #             logger.warning(f"Found {outliers_low + outliers_high} outliers in column {col}")
                
    #             # Clip outliers
    #             df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    df_clean = df_clean.sort_values(by=['store', 'date'])
    # Check and fill missing values
    missing_values = df_clean.isnull().sum()
    missing_columns = missing_values[missing_values > 0].index
    
    if not missing_columns.empty:
        logger.warning(f"Found missing values in {len(missing_columns.tolist())} columns: \n{missing_values[missing_values > 0]}")
        # Fill gaps for all stores
        df_clean[missing_columns] = (df_clean
            .groupby('store')[missing_columns]
            .transform(lambda x: x.bfill().ffill())
        )
    
    return df_clean

def aggregate_by_store_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates data by store and date, creating one unique set of records for each (store, date) pair.
    
    Args:
        df (pandas.DataFrame): Source dataframe with transaction-level data
        
    Returns:
        pandas.DataFrame: Aggregated dataframe with unique (store, date) pairs
    """
    logger.info("Aggregating data by store and date")
    df = df.sort_values(by=['store', 'date'])
    # First group by store to get data for each store
    store_attributes = df.groupby('store').agg({
        'name': 'first',
        'address': 'first',
        'city': 'first',
        'zipcode': 'first',
        'store_location': 'first',
        'county': 'first'
    }).reset_index()
    # Create list of dictionaries with category info for each store and date
    category_info = df.groupby(['store', 'date']).apply(
        lambda x: x[['category_name', 'im_desc', 'pack', 
                     'bottle_volume_ml','state_bottle_cost','state_bottle_retail',
                     'sale_dollars','sale_bottles','sale_liters','sale_gallons']]
        .drop_duplicates()
        .to_dict('records'), include_groups=False
    ).reset_index(name='item_details')
    # Merge category info with store attributes
    store_attributes = store_attributes.merge(category_info, on='store', how='left')
    # Extract coordinates from store_location
    store_locations = store_attributes['store_location'].apply(eval)
    store_attributes['lon'] = store_locations.apply(lambda x: x['coordinates'][0])
    store_attributes['lat'] = store_locations.apply(lambda x: x['coordinates'][1])
    
    logger.info(f"Obtained attributes for {len(store_attributes)} unique stores")
    
    # Now group by store and date to aggregate sales and other metrics
    aggregated = df.groupby(['store', 'date']).agg({
        # Sales - basic metrics and distributions
        'sale_bottles': ['sum', 'mean', 'median', 'std', 'min', 'max', 'skew', 'sem'],
        'sale_dollars': ['sum', 'mean', 'median', 'std', 'min', 'max', 'skew', 'sem'],
        'sale_liters': ['sum', 'mean', 'median', 'std', 'min', 'max', 'skew', 'sem'],
        'sale_gallons': ['sum', 'mean', 'median', 'std', 'min', 'max'],
        
        # Number of transactions and unique receipts
        'invoice_line_no': ['count', 'nunique'],
        
        # Statistics on bottle costs
        'state_bottle_cost': ['mean', 'median', 'std', 'min', 'max'],
        'state_bottle_retail': ['mean', 'median', 'std', 'min', 'max'],
        
        # Statistics on categories and items
        'category': ['nunique', 'count'],
        'category_name': ['nunique', 'count'],
        'itemno': ['nunique', 'count'],
        
        # Statistics on packs and volumes
        'pack': ['mean', 'median', 'std', 'min', 'max', 'sum'],
        'bottle_volume_ml': ['mean', 'median', 'std', 'min', 'max', 'sum']
    }).reset_index()
    
    # Collapse multiindex columns into single level
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    
    # Rename columns for clarity
    aggregated = aggregated.rename(columns={
        'invoice_line_no_count': 'transaction_count',
        'invoice_line_no_nunique': 'unique_transactions',
        'category_nunique': 'unique_categories',
        'category_name_nunique': 'unique_category_names',
        'itemno_nunique': 'unique_items',
        'sale_dollars_sum': 'sale_dollars',
        'sale_bottles_sum': 'sale_bottles',
        'sale_liters_sum': 'sale_liters',
        'sale_gallons_sum': 'sale_gallons'
    })
    
    # Add store information from store_attributes
    aggregated = pd.merge(store_attributes, aggregated, on=['store', 'date'], how='left')
    
    # Create additional metrics
    aggregated['avg_price_per_bottle'] = aggregated['sale_dollars'] / aggregated['sale_bottles'].replace(0, np.nan)
    aggregated['avg_price_per_liter'] = aggregated['sale_dollars'] / aggregated['sale_liters'].replace(0, np.nan)
    aggregated['avg_items_per_transaction'] = aggregated['unique_items'] / aggregated['transaction_count'].replace(0, np.nan)
    
    # Fill missing values
    aggregated = aggregated.fillna({
        'avg_price_per_bottle': aggregated['avg_price_per_bottle'].median(),
        'avg_price_per_liter': aggregated['avg_price_per_liter'].median(),
        'avg_items_per_transaction': aggregated['avg_items_per_transaction'].median()
    })
    
    logger.info(f"Data aggregated to {len(aggregated)} unique (store, date) pairs")
    
    return aggregated


def prepare_dataset(input_path, output_path, target_column='sale_dollars', log_level=logging.INFO, country_code='US'):
    """
    Main data preparation function.
    
    Args:
        input_path (str): Path to source data
        output_path (str): Path to save result
        target_column (str): Target variable for analysis
        log_level (int): Logging level
        country_code (str): Country code for holidays (ISO 3166-1 alpha-2)
    """
    global logger
    logger = setup_logger(log_level)
    
    logger.info("Starting data preparation")
    # Load data
    df = load_data(input_path)
    # Aggregate data by store and date (new step)
    logger.info("Performing data aggregation by store and date")
    df_aggregated = aggregate_by_store_date(df)
    # Feature creation
    logger.info("Starting feature creation based on aggregated data")
    df_with_basic = create_basic_features(df_aggregated, country_code=country_code)
    df_with_temporal = create_temporal_features(df_with_basic, target=target_column)
    
    df_with_store = create_store_features(df_with_temporal)
    
    # Data cleaning and validation
    df_cleaned = clean_and_validate_data(df_with_store)
    df_cleaned = df_cleaned.set_index('store').sort_values(['store','date'])
        
    logger.info(f"Saving results to {output_path}")
    df_cleaned.to_parquet(output_path)
    
    logger.info(f"Data preparation completed. Dataset saved to {output_path}")
    logger.info(f"Final dataset shape: {df_cleaned.shape}")
    
    return df_cleaned

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data preparation for sales decline forecasting')
    parser.add_argument('--input', type=str, default='data/sazerac_df.csv',
                        help='Path to source data file')
    parser.add_argument('--output', type=str, default='data/sales_features.parquet',
                        help='Path to save processed data')
    parser.add_argument('--target', type=str, default='sale_dollars',
                        help='Target variable for analysis')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--country', type=str, default='US',
                        help='Country code for holidays (ISO 3166-1 alpha-2)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper())
    prepare_dataset(args.input, args.output, args.target, log_level, args.country) 