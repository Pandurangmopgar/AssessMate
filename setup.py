"""
Setup script for SHL Assessment Recommendation System.
This script scrapes the SHL catalog, generates embeddings, and builds the FAISS index.
"""
import os
import pandas as pd
import logging
from dotenv import load_dotenv

# Use the Selenium-based scraper instead of other scrapers
from scraper_selenium import scrape_shl_catalog
from embedding import EmbeddingGenerator
from search import build_faiss_index
from utils import save_dataframe, logger

# Load environment variables
load_dotenv()

def main():
    """
    Main setup function that:
    1. Scrapes the SHL catalog
    2. Generates embeddings for the assessments
    3. Builds and saves the FAISS index
    """
    # Step 1: Scrape the SHL catalog
    logger.info("Step 1: Scraping the SHL catalog...")
    try:
        df = scrape_shl_catalog()
        if df.empty:
            logger.error("Failed to scrape SHL catalog. Exiting.")
            return
        
        logger.info(f"Successfully scraped {len(df)} assessments from the SHL catalog")
        
        # Save the raw data
        save_dataframe(df, "shl_assessments.csv")
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return
    
    # Step 2: Generate embeddings
    logger.info("Step 2: Generating embeddings for assessments...")
    try:
        # Initialize the embedding generator
        api_key = os.getenv("GEMINI_API_KEY")
        embedding_generator = EmbeddingGenerator(api_key=api_key)
        
        # Prepare texts for embedding (combine all relevant fields)
        texts = []
        for _, row in df.iterrows():
            # Combine name, description, and test_type for richer context
            combined_text = f"{row['name']}. {row['description']}. Test type: {row['test_type']}"
            texts.append(combined_text)
        
        # Generate embeddings
        embeddings = embedding_generator.batch_generate_embeddings(texts)
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        return
    
    # Step 3: Build and save the FAISS index
    logger.info("Step 3: Building and saving FAISS index...")
    try:
        # Build the index
        search_engine = build_faiss_index(embeddings, df)
        
        # Save the index and DataFrame
        search_engine.save_index("shl_faiss_index.bin", "shl_assessments.csv")
        logger.info("Successfully built and saved FAISS index")
    except Exception as e:
        logger.error(f"Error during index building: {e}")
        return
    
    logger.info("Setup completed successfully!")

if __name__ == "__main__":
    main()
