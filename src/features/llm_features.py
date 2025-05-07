"""
LLM-specific feature engineering module.
Contains functions for creating features specifically for Large Language Models.
"""

import pandas as pd
import numpy as np
import json



def prepare_llm_prompts(df: pd.DataFrame, 
                       prediction_horizon: int = 7,
                       include_history: bool = True,
                       history_window: int = 7) -> list:
    """
    Prepare prompts for LLM model with enhanced structure and features.
    
    Args:
        df (pd.DataFrame): Input dataframe with LLM features
        prediction_horizon (int): Number of days to predict
        include_history (bool): Whether to include historical context
        history_window (int): Number of historical days to include
        
    Returns:
        list: List of prompts for LLM
    """
    # Create store descriptions
    store_desc_parts = []
    store_desc_parts.append("Store {store}")
    
    if 'city' in df.columns and 'county' in df.columns:
        store_desc_parts.append("in {city}, {county} county")
    elif 'city' in df.columns:
        store_desc_parts.append("in {city}")
    elif 'county' in df.columns:
        store_desc_parts.append("in {county} county")
        
    if 'lon' in df.columns and 'lat' in df.columns:
        store_desc_parts.append("Located at coordinates ({lon:.4f}, {lat:.4f})")
        
    store_desc_format = ". ".join(store_desc_parts) + "."
    
    # Create store descriptions
    df['store_description'] = df.apply(
        lambda x: store_desc_format.format(**{k: v for k, v in x.items() if k in store_desc_format}),
        axis=1
    )
    
    # Create holiday descriptions
    if 'is_holiday' in df.columns and 'holiday_name' in df.columns:
        df['holiday_description'] = df.apply(
            lambda x: f"Date is a holiday ({x['holiday_name']})." if x['is_holiday'] else "Date is not a holiday.",
            axis=1
        )
    else:
        df['holiday_description'] = ""
    
    # Create sales summary
    df['sales_summary'] = df.apply(
        lambda x: f"Sales: ${float(x['purchase_amount']):.2f}, "
                f"Bottles: {int(x['purchased_bottles'])}, "
                f"Liters: {float(x['purchased_liters']):.2f}.",
        axis=1
    )
    
    # Create transaction summary
    transaction_parts = []
    if 'transaction_count' in df.columns:
        transaction_parts.append("Transactions: {transaction_count}")
    if 'unique_categories' in df.columns:
        transaction_parts.append("Categories: {unique_categories}")
    if 'unique_items' in df.columns:
        transaction_parts.append("Items: {unique_items}")
        
    if transaction_parts:
        transaction_format = ". ".join(transaction_parts) + "."
        df['transaction_summary'] = df.apply(
            lambda x: transaction_format.format(**{k: int(v) if isinstance(v, np.integer) else v 
                                                 for k, v in x.items() if k in transaction_format}),
            axis=1
        )
    else:
        df['transaction_summary'] = ""
    
    # Clean up text columns
    for col in ['store_description', 'holiday_description', 'sales_summary', 'transaction_summary']:
        if col in df.columns:
            df[col] = df[col].str.replace('  ', ' ')
            df[col] = df[col].str.strip()
    
    prompts = []
    
    for store in df['store'].unique():
        store_data = df[df['store'] == store].sort_values('date')
        
        # Get the last date in the data
        last_date = store_data['date'].max()
        
        # Create base prompt with role and context
        base_prompt = (
            "###Context###\n"
            "You are an expert sales forecaster with deep knowledge of retail analytics and time series forecasting. "
            "Your task is to predict future sales based on historical data and store characteristics.\n\n"
            
            "###Store Information###\n"
            f"{store_data.iloc[-1]['store_description']}\n\n"
        )
        
        # Add item details if available
        if 'item_details' in store_data.columns:
            item_details = store_data.iloc[-1]['item_details']
            if pd.notna(item_details).any():
                base_prompt += "###Product Portfolio###\n"
                # Group items by category for better organization
                items_by_category = {}
                for item in item_details:
                    category = item.get('category_name', 'Other')
                    if category not in items_by_category:
                        items_by_category[category] = []
                    items_by_category[category].append(item)
                
                # Add items by category
                for category, items in items_by_category.items():
                    base_prompt += f"\n{category}:\n"
                    for item in items:
                        base_prompt += (
                            f"- {item.get('im_desc', 'Unknown')}\n"
                            f"  * Pack Size: {int(item.get('pack', 'N/A'))} units\n"
                            f"  * Volume: {int(item.get('bottle_volume_ml', 'N/A'))}ml\n"
                            f"  * Cost: ${float(item.get('state_bottle_cost', 'N/A')):.2f}\n"
                            f"  * Retail: ${float(item.get('state_bottle_retail', 'N/A')):.2f}\n"
                            f"  * Sales: {int(item.get('sale_bottles', 'N/A'))} bottles (${float(item.get('sale_dollars', 'N/A')):.2f})\n"
                        )
                base_prompt += "\n"
        
        # Add historical context if requested
        if include_history:
            history = store_data.tail(history_window)
            base_prompt += "###Historical Sales Data###\n"
            
            for _, row in history.iterrows():
                base_prompt += (
                    f"\nDate: {row['date'].strftime('%Y-%m-%d')}\n"
                    f"- Total Sales: ${float(row['purchase_amount']):.2f}\n"
                    f"- Bottles Sold: {int(row['purchased_bottles'])}\n"
                    f"- Volume Sold: {float(row['purchased_liters']):.2f} liters\n"
                )
                
                if pd.notna(row['holiday_description']):
                    base_prompt += f"- {row['holiday_description']}\n"
                
                if 'transaction_count' in row and pd.notna(row['transaction_count']):
                    base_prompt += f"- Transactions: {int(row['transaction_count'])}\n"
            
            base_prompt += "\n"
        
        # Add prediction request with specific format
        prediction_prompt = (
            "###Task###\n"
            f"Predict sales for the next {prediction_horizon} days starting from {last_date.strftime('%Y-%m-%d')}.\n\n"
            
            "###Requirements###\n"
            "1. Consider historical patterns, holidays, and store characteristics\n"
            "2. Account for weekly and seasonal trends\n"
            "3. Consider product portfolio and category mix\n"
            "4. Provide confidence intervals for your predictions\n\n"
            
            "###Output Format###\n"
            "Provide your predictions in the following JSON format:\n"
            "{\n"
            '  "predictions": [\n'
            '    {\n'
            '      "date": "YYYY-MM-DD",\n'
            '      "predicted_sales": XXX.XX,\n'
            '      "predicted_bottles": XXX,\n'
            '      "confidence_lower": XXX.XX,\n'
            '      "confidence_upper": XXX.XX,\n'
            '      "explanation": "Brief explanation of the prediction"\n'
            '    },\n'
            '    ...\n'
            '  ],\n'
            '  "overall_trend": "Description of the overall trend",\n'
            '  "key_factors": ["Factor 1", "Factor 2", ...],\n'
            '  "category_insights": {\n'
            '    "category_name": "Trend and insights for this category"\n'
            '  }\n'
            "}\n"
        )
        
        prompts.append({
            "store_id": int(store),
            "prompt": base_prompt + prediction_prompt,
            "last_date": last_date.strftime('%Y-%m-%d'),
            "prediction_horizon": int(prediction_horizon)
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