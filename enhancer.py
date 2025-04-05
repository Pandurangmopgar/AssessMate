"""
Module for enhancing assessment recommendations using Gemini API.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List, Any
import json

from utils import logger

# Load environment variables
load_dotenv()

class RecommendationEnhancer:
    def __init__(self, api_key: str = None):
        """
        Initialize the recommendation enhancer with the Gemini API.
        
        Args:
            api_key: Google Gemini API key. If None, will try to load from GEMINI_API_KEY env var.
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            logger.warning("No Gemini API key provided. Cannot enhance recommendations.")
            self.model = None
        else:
            try:
                # Configure the Gemini API with the provided key
                genai.configure(api_key=api_key)
                
                # Initialize the generative model
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("Successfully initialized Gemini generative model")
            except Exception as e:
                logger.error(f"Error initializing Gemini API: {e}")
                self.model = None
    
    def enhance_recommendations(self, raw_results: Dict[str, List[Dict[str, Any]]], query: str) -> Dict[str, Any]:
        """
        Enhance the raw assessment recommendations using Gemini.
        
        Args:
            raw_results: Raw assessment recommendation results
            query: The original user query (job description)
            
        Returns:
            Enhanced recommendation results with additional insights
        """
        if not self.model:
            logger.warning("Gemini model not initialized. Returning raw results.")
            return raw_results
            
        try:
            # First, sanitize the raw_results to ensure they can be serialized
            sanitized_results = self._sanitize_results(raw_results)
            
            # Extract the assessment results
            results = sanitized_results.get("results", [])
            
            if not results:
                logger.warning("No results to enhance")
                return sanitized_results
                
            # Create a prompt for Gemini
            prompt = self._create_enhancement_prompt(results, query)
            
            # Generate enhanced content
            response = self.model.generate_content(prompt)
            
            # Parse the enhanced content
            enhanced_content = self._parse_response(response.text)
            
            # Combine sanitized results with enhanced content
            enhanced_results = {
                "results": sanitized_results["results"],  # Keep the sanitized results for UI display
                "enhanced": enhanced_content  # Add enhanced content
            }
            
            # Validate the enhanced results can be serialized to JSON
            enhanced_results = self._ensure_json_serializable(enhanced_results)
            
            logger.info("Successfully enhanced recommendations with Gemini")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing recommendations: {e}")
            return self._sanitize_results(raw_results)  # Return sanitized version of raw results
    
    def _sanitize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize results to ensure they can be serialized to JSON.
        
        Args:
            results: Results to sanitize
            
        Returns:
            Sanitized results
        """
        try:
            # Try to serialize to JSON to check for any issues
            json.dumps(results)
            return results  # If no exception, return as is
        except (TypeError, ValueError, OverflowError) as e:
            logger.warning(f"Results contain non-serializable values: {e}")
            
            # Deep copy with sanitization
            if isinstance(results, dict):
                sanitized = {}
                for k, v in results.items():
                    sanitized[k] = self._sanitize_results(v)
                return sanitized
            elif isinstance(results, list):
                return [self._sanitize_results(item) for item in results]
            elif isinstance(results, (int, float)):
                # Replace any problematic float values
                if isinstance(results, float):
                    import math
                    if math.isnan(results) or math.isinf(results):
                        return 0.0
                return results
            else:
                # Convert anything else to a string representation
                return str(results)
    
    def _ensure_json_serializable(self, obj: Any) -> Any:
        """
        Ensure an object is JSON serializable.
        
        Args:
            obj: Object to ensure is serializable
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
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
    
    def _create_enhancement_prompt(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Create a prompt for Gemini to enhance the recommendations.
        
        Args:
            results: List of assessment results
            query: The original user query (job description)
            
        Returns:
            Prompt for Gemini
        """
        # Convert results to a string format for the prompt
        results_str = json.dumps(results, indent=2)
        
        prompt = f"""
        Act as an expert talent assessment advisor. I'll provide you with a job description and a list of SHL assessment recommendations from our system.

        Job Description:
        {query}

        Assessment Recommendations:
        {results_str}

        Please provide the following:
        1. A brief summary explaining why these assessments are relevant for the job description (2-3 sentences)
        2. For each assessment, provide a short explanation of why it's specifically relevant to this role (1-2 sentences per assessment)
        3. Suggest an optimal assessment sequence or bundle based on these recommendations

        Format your response as JSON with the following structure:
        {{
            "summary": "Overall explanation of the recommendations...",
            "assessment_insights": [
                {{
                    "name": "Assessment Name",
                    "relevance": "Why this assessment is relevant for the job..."
                }},
                ...
            ],
            "recommended_sequence": "Suggestion for assessment sequence or bundle..."
        }}
        """
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the Gemini response text to extract the enhanced content.
        
        Args:
            response_text: Response text from Gemini
            
        Returns:
            Parsed enhanced content
        """
        try:
            # Extract JSON from the response text
            # First, find the start and end of the JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("Could not find JSON in response")
                return {
                    "summary": "Unable to generate enhanced recommendations",
                    "assessment_insights": [],
                    "recommended_sequence": ""
                }
                
            json_str = response_text[start_idx:end_idx]
            
            # Parse the JSON
            enhanced_content = json.loads(json_str)
            
            # Ensure the content is JSON serializable
            enhanced_content = self._ensure_json_serializable(enhanced_content)
            
            return enhanced_content
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            # Return a default structure
            return {
                "summary": "Unable to generate enhanced recommendations",
                "assessment_insights": [],
                "recommended_sequence": ""
            }

def enhance_recommendations(raw_results: Dict[str, List[Dict[str, Any]]], query: str) -> Dict[str, Any]:
    """
    Enhance assessment recommendations using Gemini.
    
    Args:
        raw_results: Raw assessment recommendation results
        query: The original user query (job description)
        
    Returns:
        Enhanced recommendation results
    """
    enhancer = RecommendationEnhancer()
    return enhancer.enhance_recommendations(raw_results, query)

# For testing purposes
if __name__ == "__main__":
    # Sample results and query
    sample_results = {
        "results": [
            {
                "rank": 1,
                "similarity_score": 0.75,
                "name": "SHL Cognitive Assessments",
                "url": "https://www.shl.com/solutions/products/assessments/cognitive-assessments/",
                "description": "Measure analytical thinking and problem-solving abilities",
                "duration": "30 min",
                "remote": "Yes",
                "adaptive": "Yes",
                "test_type": "Cognitive"
            },
            {
                "rank": 2,
                "similarity_score": 0.68,
                "name": "SHL Personality Assessments",
                "url": "https://www.shl.com/solutions/products/assessments/personality-assessment/",
                "description": "Evaluate work style preferences and behavioral traits",
                "duration": "25 min",
                "remote": "Yes",
                "adaptive": "No",
                "test_type": "Personality"
            }
        ]
    }
    
    sample_query = "Data Scientist with experience in machine learning and statistics"
    
    # Enhance the recommendations
    enhanced_results = enhance_recommendations(sample_results, sample_query)
    
    # Print the enhanced results
    print(json.dumps(enhanced_results, indent=2))
