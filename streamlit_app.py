"""
Streamlit app for SHL Assessment Recommendation System.
"""
import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import json

from embedding import generate_gemini_embedding
from search import FAISSSearchEngine
from utils import format_results

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="wide",
)

# Title and description
st.title("SHL Assessment Recommendation System")
st.markdown("""
This application uses AI to recommend the most relevant SHL assessments based on job descriptions or role titles.
Enter a job role, title, or paste a full job description to get personalized assessment recommendations.
""")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None

def local_recommend(query, k=5):
    """
    Make a local recommendation without API, using the FAISS index directly.
    
    Args:
        query: Job description or role title
        k: Number of recommendations to return
        
    Returns:
        Dictionary with recommendation results
    """
    try:
        # Check if the index exists
        if not os.path.exists("shl_faiss_index.bin") or not os.path.exists("shl_assessments.csv"):
            st.error("FAISS index or assessments data not found. Please run setup.py first.")
            return None
        
        # Load the search engine
        search_engine = FAISSSearchEngine()
        search_engine.load_index("shl_faiss_index.bin", "shl_assessments.csv")
        
        # Generate embedding for the query
        query_embedding = generate_gemini_embedding(query)
        
        # Search for similar assessments
        indices, scores = search_engine.search(query_embedding, k=k)
        
        # Format the results
        results = format_results(indices, scores, search_engine.df)
        return results
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

def api_recommend(query, k=5):
    """
    Make a recommendation using the FastAPI endpoint.
    
    Args:
        query: Job description or role title
        k: Number of recommendations to return
        
    Returns:
        Dictionary with recommendation results
    """
    try:
        # Make the API request
        response = requests.get(
            "http://localhost:8000/recommend",
            params={"query": query, "k": k}
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

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
    
    # Selection for recommendation method
    method = st.radio(
        "Recommendation Method", 
        options=["Local", "API"], 
        index=0,
        help="Local uses the FAISS index directly, API uses the FastAPI endpoint"
    )
    
    # Submit button
    submit_button = st.form_submit_button(label="Get Recommendations")
    
    # Process the form submission
    if submit_button and query:
        with st.spinner("Generating recommendations..."):
            if method == "Local":
                st.session_state.results = local_recommend(query, k)
            else:
                st.session_state.results = api_recommend(query, k)

# Display the results
if st.session_state.results:
    st.subheader("Recommended Assessments")
    
    # Extract the results
    recommendations = st.session_state.results.get("results", [])
    
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
        
        # Display detailed information for each recommendation
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
                    st.markdown(rec['description'])

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
