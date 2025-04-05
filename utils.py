"""
Utility functions for the SHL Assessment Recommendation System.
"""
import logging
import os
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    Save a pandas DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filename: Name of the file to save to
    """
    try:
        df.to_csv(filename, index=False)
        logger.info(f"DataFrame successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filename}: {e}")
        raise

def load_dataframe(filename: str) -> pd.DataFrame:
    """
    Load a pandas DataFrame from CSV.
    
    Args:
        filename: Name of the file to load from
        
    Returns:
        Loaded DataFrame
    """
    try:
        if not os.path.exists(filename):
            logger.error(f"File {filename} does not exist")
            raise FileNotFoundError(f"File {filename} does not exist")
        
        df = pd.read_csv(filename)
        logger.info(f"DataFrame successfully loaded from {filename}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {filename}: {e}")
        raise

def format_results(indices: List[int], scores: List[float], df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Format the search results into a JSON-serializable dictionary.
    
    Args:
        indices: List of indices of the top matches
        scores: List of similarity scores for each match
        df: DataFrame containing the assessment data
        
    Returns:
        Dictionary with formatted results
    """
    try:
        results = []
        for i, (idx, score) in enumerate(zip(indices, scores)):
            # Get the row from the DataFrame
            row = df.iloc[idx]
            
            # Sanitize the score to ensure it's JSON compliant
            # Replace infinity, -infinity, or NaN with a valid number or string
            if not isinstance(score, (int, float)) or pd.isna(score) or np.isinf(score):
                sanitized_score = 0.0  # Default fallback value
                logger.warning(f"Invalid score value detected ({score}), replaced with {sanitized_score}")
            else:
                sanitized_score = float(score)  # Convert numpy.float32 to Python float
            
            # Format the result
            result = {
                "rank": i + 1,
                "similarity_score": sanitized_score,
                "name": row["name"],
                "url": row["url"],
                "description": row["description"],
                "duration": row["duration"],
                "remote": row["remote"],
                "adaptive": row["adaptive"],
                "test_type": row["test_type"]
            }
            
            results.append(result)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error formatting results: {e}")
        return {"error": str(e)}
