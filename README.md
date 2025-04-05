# SHL Assessment Recommendation System

A GenAI-powered system that recommends relevant SHL assessments based on job descriptions or role titles. The system uses Google's Gemini Embedding API to generate semantic embeddings and FAISS for efficient vector similarity search.

## Project Structure

```
shl_recommendation_system/
├── app.py                  # FastAPI application
├── scraper.py              # Web scraping module for SHL catalog
├── embedding.py            # Gemini embedding generation
├── search.py               # FAISS vector search
├── utils.py                # Utility functions and logging
├── setup.py                # Setup script to scrape and build index
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (for API keys)
└── README.md               # Project documentation
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root directory with the following content:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

You need a valid Google Gemini API key. If you don't have one, the system will use dummy embeddings for testing purposes.

### 3. Build the FAISS Index

Run the setup script to scrape the SHL catalog, generate embeddings, and build the FAISS index:

```bash
python setup.py
```

This script will:
- Scrape assessment data from the SHL product catalog
- Generate embeddings for each assessment using Gemini API
- Build and save a FAISS index for fast similarity search

### 4. Start the API Server

```bash
uvicorn app:app --reload
```

This will start the FastAPI server at http://localhost:8000.

### 5. Start the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

This will start the Streamlit web interface at http://localhost:8501.

## API Endpoints

- `GET /recommend?query={query}&k={k}` - Get recommendations based on a query
- `GET /health` - Health check endpoint
- `GET /` - Welcome endpoint

## Components

### Scraper

The `scraper.py` module contains a web scraper for the SHL product catalog. It extracts the following information for each assessment:
- Assessment Name
- Product URL
- Description/Metadata
- Duration
- Remote Testing Support
- Adaptive/IRT Support
- Test Type

### Embedding Generator

The `embedding.py` module handles generating embeddings using Google's Gemini API. It includes a fallback to dummy embeddings if no API key is provided.

### FAISS Search Engine

The `search.py` module provides a vector search engine using FAISS. It can build, save, load, and search a FAISS index.

### FastAPI Application

The `app.py` module provides a REST API for the recommendation system. It loads the pre-built FAISS index and serves recommendation requests.

### Streamlit Frontend

The `streamlit_app.py` module provides a user-friendly web interface for interacting with the recommendation system.

## Usage

1. Enter a job description or role title in the Streamlit interface
2. Select the number of recommendations to return (1-10)
3. Choose whether to use the local FAISS index or the API
4. Click "Get Recommendations"
5. View the recommended assessments and their details

## Error Handling

The system includes comprehensive error handling and logging:
- Logs are written to both the console and a file (`app.log`)
- Failed API calls fall back to dummy embeddings
- The API returns appropriate HTTP status codes and error messages

## Notes

- For production deployment, consider using a more robust web server like Gunicorn
- Add authentication mechanisms for the API
- Implement rate limiting for production use
- Consider implementing a caching mechanism for frequently used queries
