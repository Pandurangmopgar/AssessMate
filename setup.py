"""
Setup script for SHL Assessment Recommendation System.
This script scrapes the SHL catalog, generates embeddings, and builds the FAISS index.
"""
import os
import pandas as pd
import time
import logging
import re
from dotenv import load_dotenv

# Import the scraper and other necessary modules
from scraper_selenium import SHLScraper
from embedding import EmbeddingGenerator
from search import build_faiss_index
from utils import logger

def clean_scraped_data(df):
    """Clean the scraped data to remove cookie consent messages and improve quality."""
    logger.info("Cleaning scraped data...")
    
    # Function to clean a field
    def clean_field(value):
        if pd.isna(value) or not value:
            return "Not specified"
        
        value = str(value).strip()
        # Remove cookie consent messages
        if any(x in value.lower() for x in ["cookie", "permission", "consent"]):
            return "Not specified"
        
        return value
    
    # Clean all text fields
    for col in ["description", "duration", "remote", "adaptive", "test_type"]:
        df[col] = df[col].apply(clean_field)
    
    # Extract duration values from text where possible
    def extract_duration(text):
        if text == "Not specified":
            return text
            
        # Try to extract minutes value
        pattern = r'(\d+)\s*(?:min|minutes)'
        match = re.search(pattern, text.lower())
        if match:
            return f"{match.group(1)} minutes"
        
        return text
    
    df["duration"] = df["duration"].apply(extract_duration)
    
    # Remove duplicate entries
    df.drop_duplicates(subset=["name", "url"], keep="first", inplace=True)
    
    # Remove entries with meaningless names
    df = df[~df["name"].str.contains("Find assessments", case=False)]
    df = df[~df["name"].str.contains("Unknown", case=False)]
    
    logger.info(f"Data cleaning complete. {len(df)} assessments remaining.")
    return df

def setup():
    """Set up the SHL Assessment Recommendation System."""
    logger.info("Starting setup process...")
    
    # Step 1: Scrape SHL assessments
    logger.info("Step 1: Scraping SHL assessments...")
    
    # Check if we want to use existing data
    use_existing = False
    if os.path.exists("shl_assessments.csv"):
        response = input("Found existing shl_assessments.csv. Use it instead of scraping again? (y/n): ")
        use_existing = response.lower() in ["y", "yes"]
    
    if use_existing:
        logger.info("Using existing data from shl_assessments.csv")
        df = pd.read_csv("shl_assessments.csv")
    else:
        try:
            # Scrape fresh data
            scraper = SHLScraper()
            df = scraper.scrape_assessments()  # Using the improved scrape_assessments method
            
            # Clean the data
            df = clean_scraped_data(df)
            
            # Save the scraped data
            df.to_csv("shl_assessments.csv", index=False)
            logger.info(f"Saved {len(df)} assessments to shl_assessments.csv")
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return
    
    # Step 2: Generate embeddings and build FAISS index
    logger.info("Step 2: Building FAISS index...")
    
    # Load environment variables for API keys
    load_dotenv()
    
    try:
        # Initialize the embedding generator
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("No Gemini API key found. Please set one in the .env file.")
            return
            
        embedding_generator = EmbeddingGenerator(api_key=api_key)
        
        # Prepare texts for embedding (combine all relevant fields)
        texts = []
        for _, row in df.iterrows():
            # Combine name, description, and test_type for richer context
            combined_text = f"{row['name']}. {row['description']}. Test type: {row['test_type']}"
            texts.append(combined_text)
        
        # Prepare texts for embedding (combine all relevant fields)
        texts = []
        for _, row in df.iterrows():
            # Combine name, description, and test_type for richer context
            combined_text = f"{row['name']}. {row['description']}. Test type: {row['test_type'] if 'test_type' in row else 'Unknown'}"
            texts.append(combined_text)
        
        # Generate embeddings
        embeddings = embedding_generator.batch_generate_embeddings(texts)
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        
        # Build the FAISS index
        build_faiss_index(embeddings, df)
        
        logger.info("Setup complete!")
    except Exception as e:
        logger.error(f"Error during index building: {e}")

if __name__ == "__main__":
    setup()
