"""
Module for vector search using FAISS.
"""
import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import pickle
import logging

from utils import logger
from embedding import EmbeddingGenerator

class FAISSSearchEngine:
    def __init__(self, dimension: int = 768):
        """
        Initialize the FAISS search engine.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
        self.index = None
        self.df = None
    
    def build_index(self, embeddings: List[List[float]], df: pd.DataFrame) -> None:
        """
        Build a FAISS index from a list of embeddings.
        
        Args:
            embeddings: List of embeddings to index
            df: DataFrame containing the assessment data corresponding to the embeddings
        """
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Create and train the index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity (cosine if normalized)
            self.index.add(embeddings_array)
            
            # Store the DataFrame
            self.df = df
            
            logger.info(f"Successfully built FAISS index with {len(embeddings)} embeddings of dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise
    
    def save_index(self, index_path: str, df_path: str) -> None:
        """
        Save the FAISS index and DataFrame.
        
        Args:
            index_path: Path to save the index to
            df_path: Path to save the DataFrame to
        """
        try:
            if self.index is None:
                logger.error("Cannot save index: No index has been built")
                return
            
            # Save the index
            faiss.write_index(self.index, index_path)
            
            # Save the DataFrame
            self.df.to_csv(df_path, index=False)
            
            logger.info(f"Successfully saved FAISS index to {index_path} and DataFrame to {df_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def load_index(self, index_path: str, df_path: str) -> None:
        """
        Load the FAISS index and DataFrame.
        
        Args:
            index_path: Path to load the index from
            df_path: Path to load the DataFrame from
        """
        try:
            if not os.path.exists(index_path) or not os.path.exists(df_path):
                logger.error(f"Index file {index_path} or DataFrame file {df_path} does not exist")
                raise FileNotFoundError(f"Index file {index_path} or DataFrame file {df_path} does not exist")
            
            # Load the index
            self.index = faiss.read_index(index_path)
            
            # Load the DataFrame
            self.df = pd.read_csv(df_path)
            
            # Update dimension
            self.dimension = self.index.d
            
            logger.info(f"Successfully loaded FAISS index from {index_path} and DataFrame from {df_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise
    
    def search(self, query_embedding: List[float], k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Search the index for the k nearest neighbors.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            Tuple of (indices, distances)
        """
        try:
            if self.index is None:
                logger.error("Cannot search: No index has been built or loaded")
                raise ValueError("No index has been built or loaded")
            
            # Convert query embedding to numpy array
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search the index
            distances, indices = self.index.search(query_array, k)
            
            # Convert to Python lists
            indices_list = indices[0].tolist()
            distances_list = distances[0].tolist()
            
            logger.info(f"Successfully searched FAISS index for {k} nearest neighbors")
            return indices_list, distances_list
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            raise

def build_faiss_index(embedding_list: List[List[float]], df: pd.DataFrame) -> FAISSSearchEngine:
    """
    Build a FAISS index from a list of embeddings.
    
    Args:
        embedding_list: List of embeddings
        df: DataFrame containing assessment data
        
    Returns:
        Initialized FAISSSearchEngine
    """
    # Determine embedding dimension from first embedding
    if not embedding_list:
        raise ValueError("Empty embedding list")
    
    dimension = len(embedding_list[0])
    
    # Create and build the index
    search_engine = FAISSSearchEngine(dimension=dimension)
    search_engine.build_index(embedding_list, df)
    
    return search_engine

def search_index(query_embedding: List[float], search_engine: FAISSSearchEngine, k: int = 5) -> Tuple[List[int], List[float]]:
    """
    Search a FAISS index for the k nearest neighbors.
    
    Args:
        query_embedding: Query embedding
        search_engine: FAISSSearchEngine instance
        k: Number of results to return
        
    Returns:
        Tuple of (indices, similarity scores)
    """
    return search_engine.search(query_embedding, k)

# For testing purposes
if __name__ == "__main__":
    from embedding import generate_gemini_embedding
    import pandas as pd
    
    # Create a sample DataFrame
    data = {
        "name": ["Assessment 1", "Assessment 2", "Assessment 3"],
        "url": ["https://www.shl.com/1", "https://www.shl.com/2", "https://www.shl.com/3"],
        "description": [
            "A cognitive ability test measuring reasoning skills",
            "A personality assessment measuring work style preferences",
            "A situational judgment test for leadership roles"
        ],
        "duration": ["30 min", "45 min", "60 min"],
        "remote": ["Yes", "Yes", "No"],
        "adaptive": ["Yes", "No", "No"],
        "test_type": ["Cognitive", "Personality", "Situational"]
    }
    df = pd.DataFrame(data)
    
    # Generate embeddings for the descriptions
    embeddings = [generate_gemini_embedding(desc) for desc in df["description"]]
    
    # Build a FAISS index
    search_engine = build_faiss_index(embeddings, df)
    
    # Test a search
    query = "leadership assessment for managers"
    query_embedding = generate_gemini_embedding(query)
    
    indices, scores = search_index(query_embedding, search_engine, k=2)
    print(f"Search results for '{query}':")
    for i, (idx, score) in enumerate(zip(indices, scores)):
        print(f"{i+1}. {df.iloc[idx]['name']} (Score: {score:.4f})")
