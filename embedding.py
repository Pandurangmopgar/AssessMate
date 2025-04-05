"""
Module for generating embeddings using Google's Gemini API.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from typing import List, Union, Dict, Any
import logging

from utils import logger

# Load environment variables from .env file
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, api_key: str = None):
        """
        Initialize the embedding generator with the Gemini API.
        
        Args:
            api_key: Google Gemini API key. If None, will try to load from GEMINI_API_KEY env var.
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            logger.warning("No Gemini API key provided. Using dummy embedding generator.")
            self.use_dummy = True
        else:
            self.use_dummy = False
            try:
                # Configure the Gemini API with the provided key
                genai.configure(api_key=api_key)
                self.embedding_model = "models/embedding-001"  # Use the appropriate model name
                logger.info("Successfully initialized Gemini embedding model")
            except Exception as e:
                logger.error(f"Error initializing Gemini API: {e}")
                self.use_dummy = True
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text using Gemini API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of embedding values
        """
        if self.use_dummy:
            return self._generate_dummy_embedding(text)
        
        try:
            # Make API call to get embedding
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document",  # This could be "retrieval_document" or "retrieval_query" depending on context
            )
            
            # Extract embedding from response
            embedding = result["embedding"]
            logger.info(f"Successfully generated embedding of length {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fall back to dummy embedding in case of error
            return self._generate_dummy_embedding(text)
    
    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """
        Generate a dummy embedding for testing purposes when API key is not available.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of dummy embedding values
        """
        # Generate a deterministic but text-dependent embedding
        # This is just for testing and should be replaced with real embeddings
        np.random.seed(hash(text) % 2**32)
        embedding_size = 768  # Typical embedding size
        embedding = np.random.normal(0, 1, embedding_size).tolist()
        
        # Normalize to unit length
        norm = np.sqrt(sum(x**2 for x in embedding))
        embedding = [x / norm for x in embedding]
        
        logger.warning("Generated dummy embedding as Gemini API key is not available")
        return embedding
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings
        """
        embeddings = []
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding for text {i+1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings

def generate_gemini_embedding(text: str) -> List[float]:
    """
    Generate a Gemini embedding for the given text.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of embedding values
    """
    generator = EmbeddingGenerator()
    return generator.generate_embedding(text)

# For testing purposes
if __name__ == "__main__":
    # Test with a sample text
    sample_text = "This is a test text for generating embeddings."
    embedding = generate_gemini_embedding(sample_text)
    print(f"Generated embedding of length {len(embedding)}")
    print(f"First few values: {embedding[:5]}")
