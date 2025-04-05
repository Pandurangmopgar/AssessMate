"""
Streamlit app for SHL Assessment Recommendation System.
"""
import streamlit as st
import pandas as pd
import os
import math
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import necessary modules for recommendations
from embedding import generate_gemini_embedding, EmbeddingGenerator
from search import FAISSSearchEngine, search_index
from utils import format_results, logger
from enhancer import RecommendationEnhancer

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="wide",
)

# Initialize components
search_engine = None
embedding_generator = None
recommendation_enhancer = None

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
        st.sidebar.success("Successfully loaded FAISS index and DataFrame")
    else:
        st.sidebar.error(f"Index file {index_path} or DataFrame file {df_path} not found. ")
    
except Exception as e:
    st.sidebar.error(f"Error during initialization: {e}")

# Title and description
st.title("SHL Assessment Recommendation System")
st.markdown("""
This application uses AI to recommend the most relevant SHL assessments based on job descriptions or role titles.
Enter a job role, title, or paste a full job description to get personalized assessment recommendations.
""")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None

def ensure_json_serializable(obj):
    """
    Ensure an object is JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, bool, str, type(None))):
        return obj
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    else:
        # Convert anything else to a string
        return str(obj)

def generate_recommendations(query, k=5, enhance=True):
    """
    Generate recommendations based on a query using the FAISS index.
    
    Args:
        query: Job description or role title
        k: Number of recommendations to return
        enhance: Whether to enhance recommendations using Gemini
        
    Returns:
        Dictionary with recommendation results
    """
    try:
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        if search_engine is None or embedding_generator is None:
            raise ValueError("Search engine or embedding generator not initialized")
            
        # Generate embedding for the query
        query_embedding = embedding_generator.generate_embedding(query)
        
        # Search the index
        indices, scores = search_index(query_embedding, search_engine, k=k)
        
        # Sanitize scores
        sanitized_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                if math.isnan(score) or math.isinf(score):
                    sanitized_scores.append(0.0)
                else:
                    sanitized_scores.append(float(score))
            else:
                sanitized_scores.append(0.0)
        
        # Format the results
        results = format_results(indices, sanitized_scores, search_engine.df)
        
        # Ensure results are JSON serializable
        results = ensure_json_serializable(results)
        
        # Enhance if requested
        if enhance and recommendation_enhancer and recommendation_enhancer.model:
            try:
                logger.info("Enhancing recommendations with Gemini")
                enhanced_results = recommendation_enhancer.enhance_recommendations(results, query)
                
                # Ensure enhanced results are JSON serializable
                enhanced_results = ensure_json_serializable(enhanced_results)
                return enhanced_results
            except Exception as e:
                logger.error(f"Error enhancing recommendations: {e}")
                # Fall back to regular results if enhancement fails
                return results
        
        return results
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

# We no longer need the api_recommend function since all functionality is now built-in

# Create a form for user input
with st.form(key="recommendation_form"):
    # Input field for job description or role title
    query = st.text_area(
        "Enter job description or role title",
        height=150,
        placeholder="e.g. Senior Software Engineer responsible for designing and developing software applications..."
    )
    
    # Number of recommendations to return
    k = st.slider(
        "Number of recommendations", 
        min_value=1, 
        max_value=10, 
        value=5
    )
    
    # Submit button
    submit_button = st.form_submit_button(label="Get Recommendations")
    
    # Process the form submission
    if submit_button and query:
        with st.spinner("Generating recommendations with Gemini AI..."):
            st.session_state.results = generate_recommendations(query, k, enhance=True)

# Display the results
if st.session_state.results:
    st.subheader("Recommended Assessments")
    
    # Extract the results
    recommendations = st.session_state.results.get("results", [])
    enhanced_content = st.session_state.results.get("enhanced", None)
    
    if not recommendations:
        st.warning("No recommendations found.")
    else:
        # Create a DataFrame from the recommendations
        df = pd.DataFrame(recommendations)
        
        # Display a table with the recommendations
        st.dataframe(
            df[["rank", "name", "test_type", "duration", "remote", "adaptive"]],
            hide_index=True,
            use_container_width=True
        )
        
        # Display Gemini-enhanced insights if available
        if enhanced_content:
            st.subheader("Gemini AI Insights")
            
            # Display the summary
            st.markdown(f"**Summary**: {enhanced_content.get('summary', '')}")
            
            # Display the recommended sequence
            st.markdown(f"**Recommended Assessment Sequence**: {enhanced_content.get('recommended_sequence', '')}")
            
            # Display the assessment insights
            st.markdown("**Assessment Insights:**")
            for insight in enhanced_content.get('assessment_insights', []):
                st.markdown(f"* **{insight.get('name', '')}**: {insight.get('relevance', '')}")
            
            st.markdown("---")
        
        # Display detailed information for each recommendation
        st.subheader("Detailed Assessment Information")
        for i, rec in enumerate(recommendations):
            with st.expander(f"{i+1}. {rec['name']} (Score: {rec['similarity_score']:.4f})"):
                # Two columns for metadata and description
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**Test Type:** {rec['test_type']}")
                    st.markdown(f"**Duration:** {rec['duration']}")
                    st.markdown(f"**Remote Testing:** {rec['remote']}")
                    st.markdown(f"**Adaptive Testing:** {rec['adaptive']}")
                    st.markdown(f"**URL:** [{rec['url']}]({rec['url']})")
                
                with col2:
                    st.markdown("**Description:**")
                    st.markdown(rec['description'] if rec['description'] else "No detailed description available")

# Add some information about the application
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses Google's Gemini Embedding API and FAISS vector search to recommend SHL assessments based on job descriptions or role titles.
    
    The system works by comparing the semantic meaning of your job description with a database of SHL assessments, finding the most relevant matches.
    """)
    
    st.header("How to use")
    st.markdown("""
    1. Enter a job description or role title in the text area
    2. Select the number of recommendations you want
    3. Choose whether to use the local FAISS index or the API
    4. Click "Get Recommendations"
    """)
    
    st.header("System Status")
    if os.path.exists("shl_faiss_index.bin") and os.path.exists("shl_assessments.csv"):
        st.success("FAISS index and assessment data are available")
    else:
        st.error("FAISS index or assessment data not found. Run setup.py first.")
        
    # Try to check if the API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=1)
        if response.status_code == 200:
            st.success("API is running")
        else:
            st.warning("API is not responding correctly")
    except:
        st.warning("API is not running. Start it with 'uvicorn app:app --reload'")

if __name__ == "__main__":
    # This allows running the app directly with: streamlit run streamlit_app.py
    pass
