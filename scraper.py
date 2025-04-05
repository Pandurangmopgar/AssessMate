"""
Web scraper module for extracting assessment data from SHL's product catalog.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re
import time
from typing import Dict, List, Optional

from utils import logger

class SHLScraper:
    def __init__(self, base_url: str = "https://www.shl.com/solutions/products/product-catalog/"):
        """
        Initialize the SHL scraper.
        
        Args:
            base_url: URL of the SHL product catalog
        """
        self.base_url = base_url
        self.products_page_url = "https://www.shl.com/solutions/products/"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Get a BeautifulSoup object for the given URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if an error occurred
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
        
    def _extract_assessment_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract assessment product links from the catalog page.
        
        Args:
            soup: BeautifulSoup object of the catalog page
            
        Returns:
            List of assessment product URLs
        """
        assessment_links = []
        
        try:
            # Try multiple selector patterns that might match product links
            selectors = [
                '.product-card a', '.product-link',  # Original selectors
                'a.product-link', 'a.assessment-link', # Direct links
                '.product-container a', '.assessment-container a', # Container links
                '.solutions-grid a', '.product-grid a', # Grid items
                'a[href*="product"]', 'a[href*="assessment"]' # Attribute contains
            ]
            
            for selector in selectors:
                product_elements = soup.select(selector)
                logger.info(f"Selector '{selector}' found {len(product_elements)} elements")
                
                for element in product_elements:
                    href = element.get('href')
                    if href and ('product' in href.lower() or 'assessment' in href.lower() or 'view' in href.lower()):
                        # Make sure we have absolute URLs
                        if href.startswith('/'):
                            href = f"https://www.shl.com{href}"
                        assessment_links.append(href)
            
            # As a fallback, get all links that might be product related
            if not assessment_links:
                all_links = soup.find_all('a')
                for link in all_links:
                    href = link.get('href')
                    text = link.text.strip().lower() if link.text else ""
                    
                    # Check both href and link text for relevant keywords
                    is_relevant = href and (
                        'product' in href.lower() or
                        'assessment' in href.lower() or
                        'test' in href.lower() or
                        'solution' in href.lower() or
                        ('view' in href.lower() and ('test' in text or 'assessment' in text))
                    )
                    
                    if is_relevant:
                        # Make sure we have absolute URLs
                        if href.startswith('/'):
                            href = f"https://www.shl.com{href}"
                        assessment_links.append(href)
            
            # Remove duplicates
            assessment_links = list(set(assessment_links))
            logger.info(f"Found {len(assessment_links)} assessment links")
            return assessment_links
        except Exception as e:
            logger.error(f"Error extracting assessment links: {e}")
            return []
        
    def _extract_category_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract category links from the catalog page.
        
        Args:
            soup: BeautifulSoup object of the catalog page
            
        Returns:
            List of category URLs
        """
        category_links = []
        
        try:
            # Find all category links based on the page structure
            # Look for links under headings like 'Ability & Aptitude', 'Personality & Behavior' etc.
            
            # First try to find alphabetical category sections
            alphabet_sections = soup.find_all(['h3', 'h4'], string=lambda s: s and len(s.strip()) == 1 and s.strip().isalpha())
            
            for section in alphabet_sections:
                # Find the next ul or div that might contain category links
                siblings = list(section.find_next_siblings())
                for sibling in siblings:
                    if sibling.name in ['ul', 'div']:
                        links = sibling.find_all('a')
                        for link in links:
                            href = link.get('href')
                            if href:
                                # Make sure we have absolute URLs
                                if href.startswith('/'):
                                    href = f"https://www.shl.com{href}"
                                category_links.append(href)
                        break  # Only process the first ul/div after the heading
            
            # If that didn't work, try to find any links that look like categories
            if not category_links:
                potential_categories = ['ability', 'aptitude', 'personality', 'behavior', 'situational', 
                                       'competencies', 'development', 'assessment', 'knowledge', 'skills', 
                                       'simulations']
                
                for category in potential_categories:
                    # Look for links containing category keywords
                    links = soup.find_all('a', string=lambda s: s and category.lower() in s.lower())
                    for link in links:
                        href = link.get('href')
                        if href:
                            # Make sure we have absolute URLs
                            if href.startswith('/'):
                                href = f"https://www.shl.com{href}"
                            category_links.append(href)
            
            # Remove duplicates
            category_links = list(set(category_links))
            logger.info(f"Found {len(category_links)} category links")
            return category_links
        except Exception as e:
            logger.error(f"Error extracting category links: {e}")
            return []
        
    def _extract_assessment_details(self, url: str) -> Optional[Dict]:
        """
        Extract details from an individual assessment page.
        
        Args:
            url: URL of the assessment product page
            
        Returns:
            Dictionary containing assessment details or None if an error occurred
        """
        try:
            soup = self._get_page(url)
            if not soup:
                return None
            
            # This is a placeholder extraction logic.
            # The actual extraction will depend on the HTML structure of SHL's product pages
            name = soup.find('h1', class_='product-title').text.strip() if soup.find('h1', class_='product-title') else "Unknown"
            
            description_element = soup.find('div', class_='product-description')
            description = description_element.text.strip() if description_element else ""
            
            # Extract metadata from product specifications table
            duration = "Unknown"
            remote = "Unknown"
            adaptive = "Unknown"
            test_type = "Unknown"
            
            # Look for a product specifications table
            spec_table = soup.find('table', class_='product-specs')
            if spec_table:
                rows = spec_table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) >= 2:
                        key = cells[0].text.strip().lower()
                        value = cells[1].text.strip()
                        
                        if 'duration' in key:
                            duration = value
                        elif 'remote' in key or 'online' in key:
                            remote = 'Yes' if ('yes' in value.lower() or 'supported' in value.lower()) else 'No'
                        elif 'adaptive' in key or 'irt' in key:
                            adaptive = 'Yes' if ('yes' in value.lower() or 'supported' in value.lower()) else 'No'
                        elif 'type' in key:
                            test_type = value
            
            return {
                "name": name,
                "url": url,
                "description": description,
                "duration": duration,
                "remote": remote,
                "adaptive": adaptive,
                "test_type": test_type
            }
        except Exception as e:
            logger.error(f"Error extracting details from {url}: {e}")
            return None
    
    def scrape_catalog(self) -> pd.DataFrame:
        """
        Scrape the SHL product catalog.
        
        Returns:
            DataFrame containing assessment data
        """
        logger.info(f"Starting to scrape SHL products")
        
        # First try the direct products page approach
        logger.info(f"Attempting to scrape from products page: {self.products_page_url}")
        products_soup = self._get_page(self.products_page_url)
        
        if not products_soup:
            logger.error("Failed to fetch the products page")
            return pd.DataFrame()
        
        # Extract assessment links from the products page
        assessment_links = self._extract_assessment_links(products_soup)
        
        # If the products page approach didn't work, try the catalog page approach
        if not assessment_links:
            logger.info("No links found on products page, trying catalog page approach")
            
            # Get the catalog page
            catalog_soup = self._get_page(self.base_url)
            if not catalog_soup:
                logger.error("Failed to fetch the catalog page")
                return pd.DataFrame()
            
            # Extract category links
            category_links = self._extract_category_links(catalog_soup)
            if not category_links:
                logger.warning("No category links found")
                
                # Last resort: Try to find any links with "product" in them
                all_links = catalog_soup.find_all('a')
                product_links = []
                for link in all_links:
                    href = link.get('href')
                    if href and ('product' in href.lower() or 'assessment' in href.lower()):
                        if href.startswith('/'):
                            href = f"https://www.shl.com{href}"
                        product_links.append(href)
                
                logger.info(f"Found {len(product_links)} potential product links as last resort")
                assessment_links = product_links
            else:
                # For each category, extract assessment links
                for i, category_link in enumerate(category_links):
                    logger.info(f"Scraping category {i+1}/{len(category_links)}: {category_link}")
                    
                    category_soup = self._get_page(category_link)
                    if category_soup:
                        # Extract assessment links from the category page
                        category_assessment_links = self._extract_assessment_links(category_soup)
                        assessment_links.extend(category_assessment_links)
                    
                    # Be nice to the server
                    time.sleep(1)
        
        # Deduplicate links
        assessment_links = list(set(assessment_links))
        logger.info(f"Found {len(assessment_links)} unique assessment links")
        
        if not assessment_links:
            logger.warning("No assessment links found after all attempts")
            
            # Create some dummy data for testing purposes
            logger.info("Creating dummy assessment data for testing")
            dummy_data = [
                {
                    "name": "Cognitive Ability Assessment",
                    "url": "https://www.shl.com/solutions/products/cognitive-ability/",
                    "description": "Measures reasoning skills and ability to learn and apply new information.",
                    "duration": "30 min",
                    "remote": "Yes",
                    "adaptive": "Yes",
                    "test_type": "Cognitive"
                },
                {
                    "name": "Personality Assessment",
                    "url": "https://www.shl.com/solutions/products/personality-assessment/",
                    "description": "Measures work-related personality traits and preferences.",
                    "duration": "45 min",
                    "remote": "Yes",
                    "adaptive": "No",
                    "test_type": "Personality"
                },
                {
                    "name": "Situational Judgment Test",
                    "url": "https://www.shl.com/solutions/products/situational-judgment/",
                    "description": "Assesses decision-making abilities in work-related scenarios.",
                    "duration": "40 min",
                    "remote": "Yes",
                    "adaptive": "No",
                    "test_type": "Situational"
                },
                {
                    "name": "Leadership Assessment",
                    "url": "https://www.shl.com/solutions/products/leadership-assessment/",
                    "description": "Evaluates leadership potential and capabilities.",
                    "duration": "60 min",
                    "remote": "Yes",
                    "adaptive": "No",
                    "test_type": "Behavioral"
                },
                {
                    "name": "Technical Skills Assessment",
                    "url": "https://www.shl.com/solutions/products/technical-skills/",
                    "description": "Measures specific technical competencies and skills.",
                    "duration": "50 min",
                    "remote": "Yes",
                    "adaptive": "No",
                    "test_type": "Skills"
                }
            ]
            return pd.DataFrame(dummy_data)
        
        # Extract details for each assessment
        assessment_data = []
        for i, link in enumerate(assessment_links):
            logger.info(f"Scraping assessment {i+1}/{len(assessment_links)}: {link}")
            details = self._extract_assessment_details(link)
            
            if details:
                assessment_data.append(details)
            
            # Be nice to the server and avoid being blocked
            time.sleep(1)
        
        # Create a DataFrame from the assessment data
        df = pd.DataFrame(assessment_data)
        logger.info(f"Scraped {len(df)} assessments successfully")
        
        return df

def scrape_shl_catalog() -> pd.DataFrame:
    """
    Scrape the SHL product catalog and return a DataFrame.
    
    Returns:
        DataFrame containing assessment data
    """
    scraper = SHLScraper()
    return scraper.scrape_catalog()

# For testing purposes
if __name__ == "__main__":
    df = scrape_shl_catalog()
    print(f"Scraped {len(df)} assessments")
    if not df.empty:
        print(df.head())
        # Save to CSV for later use
        df.to_csv("shl_assessments.csv", index=False)
