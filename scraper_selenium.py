"""
Selenium-based web scraper module for extracting assessment data from SHL's product catalog.
"""
import pandas as pd
import time
import os
from typing import Dict, List, Optional
import logging
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

from utils import logger

class SHLScraper:
    def __init__(self, base_url: str = "https://www.shl.com/solutions/products/"):
        """
        Initialize the SHL scraper with Selenium.
        
        Args:
            base_url: URL of the SHL products page
        """
        self.base_url = base_url
        self.product_catalog_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.assessments_url = "https://www.shl.com/solutions/products/assessments/"
        
        # Setup headless Chrome browser
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            # Initialize the Chrome WebDriver with ChromeDriverManager
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("Successfully initialized Selenium WebDriver")
        except Exception as e:
            logger.error(f"Error initializing Selenium WebDriver: {e}")
            raise
    
    def __del__(self):
        """Close the browser when the object is destroyed."""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                logger.info("WebDriver closed")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {e}")
    
    def _get_page_content(self, url: str) -> Optional[str]:
        """
        Get the HTML content of a page.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if an error occurred
        """
        try:
            logger.info(f"Navigating to: {url}")
            self.driver.get(url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Add a small delay to ensure JavaScript content is loaded
            time.sleep(2)
            
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
    
    def _get_assessment_links(self) -> List[str]:
        """
        Extract assessment product links from the SHL website.
        
        Returns:
            List of assessment product URLs
        """
        logger.info(f"Extracting assessment links from SHL website")
        
        # Define multiple starting points for better coverage
        urls_to_check = [
            self.base_url,  # Main products page
            self.product_catalog_url,  # Product catalog page
            self.assessments_url  # Assessments page
        ]
        
        all_assessment_links = set()
        
        for url in urls_to_check:
            html_content = self._get_page_content(url)
            if not html_content:
                continue
                
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try different selectors to find product/assessment links
            selectors = [
                "a[href*='assessment']", 
                "a[href*='product']", 
                ".product-card a", 
                ".solution-card a",
                ".card a",
                "a.card-link",
                "a.product-link"
            ]
            
            for selector in selectors:
                try:
                    # Using Selenium to find elements
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    logger.info(f"Selector '{selector}' found {len(elements)} elements")
                    
                    for element in elements:
                        href = element.get_attribute("href")
                        if href and self._is_assessment_link(href):
                            all_assessment_links.add(href)
                except Exception as e:
                    logger.error(f"Error with selector {selector}: {e}")
            
            # Find all links using BeautifulSoup as well for better coverage
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Make sure we have absolute URLs
                if href.startswith('/'):
                    href = f"https://www.shl.com{href}"
                
                if self._is_assessment_link(href):
                    all_assessment_links.add(href)
        
        # Convert to list and log results
        assessment_links = list(all_assessment_links)
        
        # Filter out duplicates and non-assessment pages
        unique_links = []
        seen_paths = set()
        
        for link in assessment_links:
            # Extract the path part of the URL
            path = link.split("shl.com")[-1] if "shl.com" in link else link
            
            # Skip if we've seen this path before
            if path in seen_paths:
                continue
                
            seen_paths.add(path)
            unique_links.append(link)
        
        logger.info(f"Found {len(unique_links)} unique assessment links")
        return unique_links
    
    def _is_assessment_link(self, url: str) -> bool:
        """
        Check if a URL is likely to be an assessment product page.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is likely an assessment page, False otherwise
        """
        # Skip external links
        if not ("shl.com" in url or url.startswith("/")):
            return False
            
        # Skip irrelevant pages
        if any(skip in url.lower() for skip in [
            "/blog/", 
            "/news/", 
            "/contact", 
            "/about", 
            "/login", 
            "/resources/blog",
            "/privacy",
            "/terms",
            ".pdf",
            ".doc",
            ".jpg",
            ".png"
        ]):
            return False
            
        # Include if it contains assessment-related keywords
        return any(keyword in url.lower() for keyword in [
            "assessment",
            "test",
            "/products/",
            "personality",
            "cognitive",
            "ability",
            "skill",
            "simulation",
            "behavioral",
            "situational",
            "judgment"
        ])
    
    def _extract_assessment_details(self, url: str) -> Optional[Dict]:
        """
        Extract details from an individual assessment page.
        
        Args:
            url: URL of the assessment product page
            
        Returns:
            Dictionary containing assessment details or None if an error occurred
        """
        try:
            html_content = self._get_page_content(url)
            if not html_content:
                return None
                
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract name (try different selectors)
            name = "Unknown"
            name_selectors = ["h1", ".product-title", ".page-title", ".hero-title", ".title"]
            for selector in name_selectors:
                element = soup.select_one(selector)
                if element and element.text.strip():
                    name = element.text.strip()
                    break
            
            # Extract description
            description = ""
            desc_selectors = [
                ".product-description", 
                ".description", 
                ".overview", 
                "section p", 
                ".content p"
            ]
            
            for selector in desc_selectors:
                elements = soup.select(selector)
                if elements:
                    # Combine the first few paragraphs
                    description = " ".join(
                        [e.text.strip() for e in elements[:3] if e.text.strip()]
                    )
                    break
            
            # Extract metadata
            duration = self._extract_metadata(soup, ["duration", "time", "length"])
            remote = self._extract_yes_no(soup, ["remote", "online", "virtual"])
            adaptive = self._extract_yes_no(soup, ["adaptive", "irt", "item response", "tailored"])
            test_type = self._extract_test_type(soup)
            
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
    
    def _extract_metadata(self, soup: BeautifulSoup, keywords: List[str]) -> str:
        """
        Extract metadata that matches any of the given keywords.
        
        Args:
            soup: BeautifulSoup object
            keywords: List of keywords to look for
            
        Returns:
            Extracted value or "Unknown"
        """
        # Check in table rows
        for tr in soup.find_all('tr'):
            th_text = tr.find('th').text.lower() if tr.find('th') else ""
            td = tr.find('td')
            
            if td and any(keyword in th_text for keyword in keywords):
                return td.text.strip()
        
        # Check in definition lists
        for dt in soup.find_all('dt'):
            dt_text = dt.text.lower()
            dd = dt.find_next('dd')
            
            if dd and any(keyword in dt_text for keyword in keywords):
                return dd.text.strip()
        
        # Check in paragraph text
        for p in soup.find_all('p'):
            p_text = p.text.lower()
            for keyword in keywords:
                match = re.search(f"{keyword}:?\s*([^\.]+)", p_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return "Unknown"
    
    def _extract_yes_no(self, soup: BeautifulSoup, keywords: List[str]) -> str:
        """
        Extract a Yes/No value based on keywords.
        
        Args:
            soup: BeautifulSoup object
            keywords: List of keywords to look for
            
        Returns:
            "Yes", "No", or "Unknown"
        """
        value = self._extract_metadata(soup, keywords)
        
        if value != "Unknown":
            if any(pos in value.lower() for pos in ["yes", "true", "supported", "available"]):
                return "Yes"
            elif any(neg in value.lower() for neg in ["no", "false", "not supported", "unavailable"]):
                return "No"
                
        # Search for the keywords in the entire page
        page_text = soup.get_text().lower()
        for keyword in keywords:
            if f"{keyword} is supported" in page_text or f"{keyword} available" in page_text:
                return "Yes"
            if f"{keyword} is not supported" in page_text or f"{keyword} not available" in page_text:
                return "No"
                
        return "Unknown"
    
    def _extract_test_type(self, soup: BeautifulSoup) -> str:
        """
        Extract the test type from the page.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Test type or "Unknown"
        """
        # Check for test type in metadata
        test_type = self._extract_metadata(soup, ["type", "category", "assessment type"])
        if test_type != "Unknown":
            return test_type
            
        # Look for common test type keywords in the page
        page_text = soup.get_text().lower()
        test_types = [
            "Cognitive", "Personality", "Behavioral", "Situational Judgment",
            "Skills", "Simulations", "Language", "Technical", "Coding", "Leadership",
            "Verbal", "Numerical", "Logical", "Inductive", "Assessment Center"
        ]
        
        # Check URL first
        url = soup.url if hasattr(soup, 'url') else ""
        for t_type in test_types:
            if t_type.lower() in url.lower():
                return t_type
                
        # Check page content
        for t_type in test_types:
            if t_type.lower() in page_text:
                context = self._get_context(page_text, t_type.lower(), 100)
                if "assessment" in context or "test" in context:
                    return t_type
                    
        # Look for H1 or title with type information
        h1 = soup.find('h1')
        if h1:
            for t_type in test_types:
                if t_type.lower() in h1.text.lower():
                    return t_type
                    
        return "Unknown"
    
    def _get_context(self, text: str, keyword: str, chars: int) -> str:
        """
        Get context around a keyword in text.
        
        Args:
            text: Text to search in
            keyword: Keyword to find
            chars: Number of characters of context to extract
            
        Returns:
            Context around the keyword
        """
        idx = text.find(keyword)
        if idx == -1:
            return ""
            
        start = max(0, idx - chars)
        end = min(len(text), idx + len(keyword) + chars)
        
        return text[start:end]
    
    def scrape_catalog(self) -> pd.DataFrame:
        """
        Scrape the SHL assessment catalog.
        
        Returns:
            DataFrame containing assessment data
        """
        logger.info("Starting to scrape SHL products")
        
        try:
            # Get assessment links
            assessment_links = self._get_assessment_links()
            
            if not assessment_links:
                logger.warning("No assessment links found")
                return pd.DataFrame()
            
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
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return pd.DataFrame()
        finally:
            try:
                self.driver.quit()
                logger.info("WebDriver closed")
            except:
                pass

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
