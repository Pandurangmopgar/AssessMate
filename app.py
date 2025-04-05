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

# Global variables for the search engine and embedding generator
search_engine = None
embedding_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine and embedding generator on startup."""
    global search_engine, embedding_generator
    
    try:
        # Initialize embedding generator
        api_key = os.getenv("GEMINI_API_KEY")
        embedding_generator = EmbeddingGenerator(api_key=api_key)
        
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
    return {"status": "ok", "message": "Service is up and running"}

@app.get("/recommend")
async def recommend(
    query: str = Query(..., description="Job description or role title"),
    k: int = Query(5, description="Number of recommendations to return", ge=1, le=10)
):
    """
    Recommend SHL assessments based on the query.
    
    Args:
        query: Job description or role title
        k: Number of recommendations to return (1-10)
        
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
        
        # Format the results
        results = format_results(indices, scores, search_engine.df)
        
        return results
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
