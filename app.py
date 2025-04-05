"""
FastAPI application for the SHL Assessment Recommendation System.
"""
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

from embedding import generate_gemini_embedding, EmbeddingGenerator
from search import FAISSSearchEngine, search_index
from utils import format_results, logger
from enhancer import RecommendationEnhancer

# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions or role titles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for the search engine, embedding generator, and recommendation enhancer
search_engine = None
embedding_generator = None
recommendation_enhancer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine, embedding generator, and recommendation enhancer on startup."""
    global search_engine, embedding_generator, recommendation_enhancer
    
    try:
        # Initialize embedding generator and recommendation enhancer
        api_key = os.getenv("GEMINI_API_KEY")
        embedding_generator = EmbeddingGenerator(api_key=api_key)
        recommendation_enhancer = RecommendationEnhancer(api_key=api_key)
        
        # Load the pre-built FAISS index and DataFrame
        index_path = "shl_faiss_index.bin"
        df_path = "shl_assessments.csv"
        
        if os.path.exists(index_path) and os.path.exists(df_path):
            search_engine = FAISSSearchEngine()
            search_engine.load_index(index_path, df_path)
            logger.info("Successfully loaded FAISS index and DataFrame on startup")
        else:
            logger.warning(f"Index file {index_path} or DataFrame file {df_path} not found. "
                         f"You need to run the indexing script first.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the SHL Assessment Recommendation API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if search_engine is None or embedding_generator is None:
        return {"status": "error", "message": "Service not fully initialized"}
    
    enhancer_status = "available" if recommendation_enhancer and recommendation_enhancer.model else "unavailable"
    return {
        "status": "ok", 
        "message": "Service is up and running",
        "enhancer": enhancer_status
    }

@app.get("/recommend")
async def recommend(
    query: str = Query(..., description="Job description or role title"),
    k: int = Query(5, description="Number of recommendations to return", ge=1, le=10),
    enhance: bool = Query(False, description="Whether to enhance recommendations using Gemini")
):
    """
    Recommend SHL assessments based on the query.
    
    Args:
        query: Job description or role title
        k: Number of recommendations to return (1-10)
        enhance: Whether to enhance recommendations using Gemini
        
    Returns:
        JSON response containing the recommendations
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        # Generate embedding for the query
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Search the index
        indices, scores = search_index(query_embedding, search_engine, k=k)
        
        # Sanitize the scores to ensure JSON compatibility
        sanitized_scores = []
        for score in scores:
            import numpy as np
            import math
            
            # Replace any invalid float values (NaN, infinity) with 0.0
            if isinstance(score, (int, float)):
                if math.isnan(score) or math.isinf(score):
                    sanitized_scores.append(0.0)
                    logger.warning(f"Replaced invalid score {score} with 0.0")
                else:
                    sanitized_scores.append(float(score))  # Convert numpy.float32 to Python float
            else:
                sanitized_scores.append(0.0)
                logger.warning(f"Replaced non-numeric score {score} with 0.0")
        
        # Format the results using sanitized scores
        results = format_results(indices, sanitized_scores, search_engine.df)
        
        # Ensure all values in results are JSON serializable
        def ensure_json_serializable(obj):
            """Make sure an object is JSON serializable"""
            if isinstance(obj, dict):
                return {k: ensure_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_json_serializable(item) for item in obj]
            elif isinstance(obj, (int, bool, str, type(None))):
                return obj
            elif isinstance(obj, float):
                import math
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return obj
            else:
                # Convert anything else to a string
                return str(obj)
        
        # Sanitize the entire results structure
        results = ensure_json_serializable(results)
        
        # Double-check the results can be serialized to JSON
        import json
        try:
            json.dumps(results)
        except (TypeError, ValueError, OverflowError) as json_err:
            logger.error(f"JSON serialization error after sanitization: {json_err}")
            # Create a minimal safe response with only essential fields
            results = {"results": [], "error": "Unable to serialize full results"}
            for i, idx in enumerate(indices):
                if i >= len(sanitized_scores):
                    break
                    
                row = search_engine.df.iloc[idx]
                safe_result = {
                    "rank": i + 1,
                    "similarity_score": 0.0,
                    "name": str(row["name"]),
                    "url": str(row["url"]),
                }
                results["results"].append(safe_result)
        
        # Enhance recommendations if requested and enhancer is available
        if enhance and recommendation_enhancer and recommendation_enhancer.model:
            try:
                logger.info("Enhancing recommendations with Gemini")
                enhanced_results = recommendation_enhancer.enhance_recommendations(results, query)
                
                # Validate JSON serialization again after enhancement
                try:
                    json.dumps(enhanced_results)
                    return enhanced_results
                except (TypeError, ValueError, OverflowError) as json_err:
                    logger.error(f"Enhanced results not serializable: {json_err}")
                    # Fall back to regular results
                    return results
            except Exception as enhance_err:
                logger.error(f"Error during enhancement: {enhance_err}")
                return results
        
        return results
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        # Return a graceful error response that can always be serialized
        return {"error": str(e), "results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
