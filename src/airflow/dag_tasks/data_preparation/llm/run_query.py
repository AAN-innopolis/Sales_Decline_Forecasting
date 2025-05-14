import json
import openai
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
from src.features.llm_features import prepare_llm_prompts

class LLMForecaster:
    def __init__(self, api_key, base_url):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "gpt-4-turbo"
    
    def generate_prompts(self, 
                        df: pd.DataFrame,
                        store_id: int,
                        last_date: str,
                        prediction_horizon: int = 7,
                        include_history: bool = True) -> List[Dict]:
        """Generate LLM prompts for a specific store"""
        return prepare_llm_prompts(
            df=df,
            store_id=store_id,
            last_date=last_date,
            prediction_horizon=prediction_horizon,
            include_history=include_history
        )
    
    def query_llm(self, prompt: str) -> Optional[Dict]:
        """Execute LLM query with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sales forecasting analyst. Please provide your response in JSON format."},
                    {"role": "user", "content": f"Please analyze the following data and provide a sales forecast in JSON format: {prompt}"}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise LLMError(f"LLM query failed: {str(e)}") from e
    
    def process_prediction(self, 
                          prediction: Dict,
                          start_date: pd.Timestamp,
                          days: int) -> pd.Series:
        """Convert LLM response to pandas Series"""
        try:
            dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=days)
            values = [pred.get('predicted_sales', None) for pred in prediction['predictions']]
            
            # Handle case where we get fewer predictions than requested
            if len(values) < days:
                print(f"Warning: LLM returned {len(values)} predictions, but {days} were requested")
                # Extend the values list to match requested days
                if len(values) > 0:
                    # If we have at least one value, repeat the last value
                    last_value = values[-1]
                    values.extend([last_value] * (days - len(values)))
                else:
                    # If no values, pad with None
                    values = [None] * days
            
            return pd.Series(values, index=dates)
        except KeyError as e:
            raise ProcessingError(f"Invalid prediction format: {str(e)}")

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

class ProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass