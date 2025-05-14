"""
LLM-specific feature engineering module.
Contains functions for creating features specifically for Large Language Models.
"""


import json
from typing import Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

def prepare_llm_prompts(df: pd.DataFrame, 
                       prediction_horizon: int = 7,
                       include_history: bool = True,
                       history_window: int = 7,
                       store_id: Optional[int] = None,
                       last_date: Optional[Union[str, datetime]] = None) -> list:
    """
    Prepare prompts for LLM model with enhanced structure and features.
    
    Args:
        df (pd.DataFrame): Input dataframe with LLM features
        prediction_horizon (int): Number of days to predict
        include_history (bool): Whether to include historical context
        history_window (int): Number of historical days to include
        store_id (int, optional): Specific store ID to process
        last_date (str/datetime, optional): Specific last date for prediction start
    
    Returns:
        list: List of prompts for LLM
    """
    # Filter by store_id if specified
    if store_id is not None:
        df = df[df['store'] == store_id]
        if df.empty:
            return []

    # Convert last_date to datetime if provided as string
    last_date_dt = None
    if last_date is not None:
        if isinstance(last_date, str):
            last_date_dt = pd.to_datetime(last_date)
        else:
            last_date_dt = last_date

    # Create store descriptions
    store_desc_parts = ["Store {store}"]
    if 'city' in df.columns and 'county' in df.columns:
        store_desc_parts.append("in {city}, {county} county")
    elif 'city' in df.columns:
        store_desc_parts.append("in {city}")
    elif 'county' in df.columns:
        store_desc_parts.append("in {county} county")
    if 'lon' in df.columns and 'lat' in df.columns:
        store_desc_parts.append("Located at coordinates ({lon:.4f}, {lat:.4f})")
    
    store_desc_format = ". ".join(store_desc_parts) + "."
    
    # Create dynamic columns
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    df.loc[:, 'store_description'] = df.apply(
        lambda x: store_desc_format.format(**x.to_dict()), axis=1
    )

    # Holiday descriptions
    if 'is_holiday' in df.columns and 'holiday_name' in df.columns:
        df.loc[:, 'holiday_description'] = df.apply(
            lambda x: f"Date is a holiday ({x['holiday_name']})." if x['is_holiday'] else "", 
            axis=1
        )
    else:
        df.loc[:, 'holiday_description'] = ""

    # Sales summary
    df.loc[:, 'sales_summary'] = df.apply(
        lambda x: f"Sales: ${x['purchase_amount']:.2f}, "
                f"Bottles: {int(x['purchased_bottles'])}, "
                f"Liters: {x['purchased_liters']:.2f}.", 
        axis=1
    )

    # Transaction summary
    transaction_parts = []
    if 'transaction_count' in df.columns:
        transaction_parts.append("Transactions: {transaction_count}")
    if 'unique_categories' in df.columns:
        transaction_parts.append("Categories: {unique_categories}")
    if 'unique_items' in df.columns:
        transaction_parts.append("Items: {unique_items}")
    
    df.loc[:, 'transaction_summary'] = ""
    if transaction_parts:
        transaction_format = ". ".join(transaction_parts) + "."
        df.loc[:, 'transaction_summary'] = df.apply(
            lambda x: transaction_format.format(**x.to_dict()), axis=1
        )

    # Clean text columns
    text_cols = ['store_description', 'holiday_description', 'sales_summary', 'transaction_summary']
    for col in text_cols:
        df.loc[:, col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()

    prompts = []
    for store in df['store'].unique():
        store_data = df[df['store'] == store].sort_values('date')
        
        # Date handling
        if last_date_dt is not None:
            if store_data['date'].max() < last_date_dt:
                continue  # Skip if store doesn't have data up to last_date
            store_data = store_data[store_data['date'] <= last_date_dt]
            current_last_date = last_date_dt
        else:
            current_last_date = store_data['date'].max()

        # Base prompt components
        base_prompt = f"""###Context###
You are an expert sales forecaster with deep knowledge of retail analytics and time series forecasting.
Your task is to predict future sales based on historical data and store characteristics.

###Store Information###
{store_data.iloc[-1]['store_description']}
"""

        # Add item details
        if 'item_details' in store_data.columns:
            try:
                items = store_data.iloc[-1]['item_details']
                # Simple check - first confirm it's iterable by trying to iterate
                if items and hasattr(items, '__iter__'):
                    base_prompt += "\n###Product Portfolio###\n"
                    items_by_category = {}
                    for item in items:
                        category = item.get('category_name', 'Other')
                        items_by_category.setdefault(category, []).append(item)
                    
                    for cat, cat_items in items_by_category.items():
                        base_prompt += f"\n{cat}:\n"
                        for item in cat_items:
                            base_prompt += (f"- {item.get('im_desc', 'Unknown')}\n"
                                          f"  * Pack Size: {item.get('pack', 'N/A')} units\n"
                                          f"  * Volume: {item.get('bottle_volume_ml', 'N/A')}ml\n")
            except:
                # Silently continue if there's any error with items
                print("")

        # Historical data
        if include_history:
            history = store_data.tail(history_window)
            base_prompt += "\n###Historical Sales Data###\n"
            for _, row in history.iterrows():
                hist_entry = (f"\nDate: {row['date'].strftime('%Y-%m-%d')}\n"
                             f"- Total Sales: ${row['purchase_amount']:.2f}\n"
                             f"- Bottles Sold: {int(row['purchased_bottles'])}\n")
                if row['holiday_description']:
                    hist_entry += f"- {row['holiday_description']}\n"
                base_prompt += hist_entry

        # Prediction task
        prediction_prompt = f"""
###Task###
Predict sales for the next {prediction_horizon} days starting from {current_last_date.strftime('%Y-%m-%d')}.

###Requirements###
1. Consider historical patterns, holidays, and store characteristics
2. Account for weekly and seasonal trends
3. Provide confidence intervals for predictions
4. YOU MUST provide predictions for EXACTLY {prediction_horizon} days

###Output Format###
{{
  "predictions": [
    {{
      "date": "YYYY-MM-DD",
      "predicted_sales": XXX.XX,
      "confidence_lower": XXX.XX,
      "confidence_upper": XXX.XX,
      "reasoning": "Brief explanation"
    }},
    ... (repeat for all {prediction_horizon} days)
  ],
  "key_factors": ["Factor1", "Factor2"]
}}"""

        prompts.append({
            "store_id": int(store),
            "prompt": base_prompt + prediction_prompt,
            "prediction_start": current_last_date.strftime('%Y-%m-%d'),
            "horizon": prediction_horizon
        })

    return prompts

def parse_llm_response(response: str) -> dict:
    """
    Parse LLM response into structured format.
    
    Args:
        response (str): LLM model response
        
    Returns:
        dict: Parsed predictions
    """
    try:
        # Try to parse as JSON first
        predictions = json.loads(response)
        return predictions
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract predictions from text
        predictions = {
            "predictions": [],
            "overall_trend": "",
            "key_factors": [],
            "category_insights": {}
        }
        
        # Split response into lines
        lines = response.strip().split('\n')
        
        current_prediction = None
        for line in lines:
            line = line.strip()
            
            # Look for date and predictions
            if 'date' in line.lower() and any(x in line.lower() for x in ['predicted', 'forecast']):
                if current_prediction:
                    predictions["predictions"].append(current_prediction)
                
                try:
                    # Extract date and predictions
                    date_str = line.split('date:')[1].split(',')[0].strip()
                    sales_str = line.split('sales:')[1].split(',')[0].strip()
                    bottles_str = line.split('bottles:')[1].strip()
                    
                    current_prediction = {
                        'date': pd.to_datetime(date_str).strftime('%Y-%m-%d'),
                        'predicted_sales': float(sales_str.replace('$', '')),
                        'predicted_bottles': int(bottles_str),
                        'confidence_lower': None,
                        'confidence_upper': None,
                        'explanation': ''
                    }
                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
            
            # Look for overall trend
            elif 'trend' in line.lower():
                predictions["overall_trend"] = line.split('trend:')[1].strip()
            
            # Look for key factors
            elif 'factor' in line.lower():
                factor = line.split('factor:')[1].strip()
                predictions["key_factors"].append(factor)
            
            # Look for category insights
            elif 'category' in line.lower():
                category_name = line.split('category:')[1].strip()
                predictions["category_insights"][category_name] = line.split('insights:')[1].strip()
        
        # Add the last prediction if exists
        if current_prediction:
            predictions["predictions"].append(current_prediction)
        
        return predictions 