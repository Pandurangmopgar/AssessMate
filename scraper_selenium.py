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
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # Setup Chrome options
        chrome_options = Options()
        # Don't run in headless mode to avoid bot detection
        # chrome_options.add_argument("--headless")  
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        # Add realistic user agent
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            # Initialize the WebDriver
            logger.info("Initializing Chrome WebDriver...")
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
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
            time.sleep(random.uniform(1, 2))
            
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
    
    def _handle_cookie_consent(self) -> None:
        """
        Handle cookie consent banners.
        """
        try:
            # Try to accept cookies to avoid consent banners
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
                ).click()
                logger.info("Accepted cookies")
                time.sleep(2)  # Wait for cookie banner to disappear
            except (TimeoutException, NoSuchElementException, ElementClickInterceptedException):
                try:
                    # Try alternative cookie button selectors
                    cookie_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Accept')]") + \
                                   self.driver.find_elements(By.XPATH, "//*[contains(text(), 'I agree')]") + \
                                   self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Accept All')]") + \
                                   self.driver.find_elements(By.XPATH, "//*[contains(@class, 'cookie') and contains(@class, 'button')]") + \
                                   self.driver.find_elements(By.ID, "onetrust-accept-btn-handler")
                    
                    for button in cookie_buttons:
                        try:
                            button.click()
                            logger.info("Clicked alternative cookie consent button")
                            time.sleep(2)  # Wait for cookie banner to disappear
                            break
                        except:
                            continue
                except:
                    logger.warning("Could not find cookie consent button")
        except Exception as e:
            logger.error(f"Error handling cookie consent: {e}")
    
    def _extract_description_selenium(self) -> str:
        """
        Extract assessment description directly using Selenium.
        
        Returns:
            Assessment description as string
        """
        try:
            # Try different selectors for descriptions
            description_selectors = [
                ".product-description", 
                ".description", 
                ".content p",
                ".product-content p",
                ".main-content p"
            ]
            
            for selector in description_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    # Combine all paragraph texts
                    texts = [el.text.strip() for el in elements if el.text.strip()]
                    if texts:
                        return " ".join(texts)
            
            return ""
        except Exception as e:
            logger.error(f"Error extracting description: {e}")
            return ""
    
    def _get_assessment_links(self) -> List[Dict]:
        """
        Extract assessment product links from the SHL website.
        
        Returns:
            List of dictionaries with assessment URLs and names
        """
        # List to store assessment links
        assessment_links = []
        
        # First, try to go to the main products page
        try:
            main_url = "https://www.shl.com/solutions/products/"
            logger.info(f"Navigating to main products page: {main_url}")
            self.driver.get(main_url)
            
            # Handle any cookie consent popups
            self._handle_cookie_consent()
            
            # Longer wait for initial page
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error accessing main page: {e}")
        
        # Let's try using known direct assessment links from the CSV as a fallback
        fallback_assessment_info = [
            {"name": "SAP ABAP (Advanced Level)", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-abap-advanced-level-new/"},
            {"name": "SAP ABAP (Intermediate Level)", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-abap-intermediate-level-new/"},
            {"name": "SAP HCM (Human Capital Management)", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-hcm-human-capital-management-new/"},
            {"name": "SAP SD (Sales and Distribution)", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-sd-sales-and-distribution-new/"},
            {"name": "SAP Hybris", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-hybris-new/"},
            {"name": "SAP Basis", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-basis-new/"},
            {"name": "SAP BW (Business Warehouse)", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-bw-business-warehouse-new/"},
            {"name": "SAP Materials Management", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-materials-management-new/"},
            {"name": "SAP Business Objects WebI", "url": "https://www.shl.com/solutions/products/product-catalog/view/sap-business-objects-webi-new/"},
            {"name": "Salesforce Development", "url": "https://www.shl.com/solutions/products/product-catalog/view/salesforce-development-new/"}
        ]
        
        # Add the fallback list to our assessment links
        assessment_links.extend(fallback_assessment_info)
        logger.info(f"Added {len(fallback_assessment_info)} fallback assessment links")
        
        # Try the normal catalog pages
        catalog_pages = [
            "https://www.shl.com/solutions/products/product-catalog/",
            "https://www.shl.com/solutions/products/product-catalog/?start=24&type=1",
            "https://www.shl.com/solutions/products/product-catalog/?start=48&type=1"
        ]
        
        for url in catalog_pages:
            try:
                logger.info(f"Navigating to catalog page: {url}")
                self.driver.get(url)
                
                # Wait for the page to load with multiple possible content selectors
                page_loaded = False
                content_selectors = [".content", ".main-content", ".product-list", ".product-catalog-component", "body"]
                
                for selector in content_selectors:
                    try:
                        WebDriverWait(self.driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        page_loaded = True
                        logger.info(f"Page loaded, found selector: {selector}")
                        break
                    except TimeoutException:
                        continue
                
                if not page_loaded:
                    logger.warning(f"Timeout waiting for content on {url}")
                    # Continue anyway and try to extract what we can
                
                # Always wait a bit to make sure JavaScript has finished running
                time.sleep(5)
                
                # Try multiple selectors for product links
                product_selectors = [
                    ".product-name a", 
                    "a[href*='product-catalog/view']", 
                    "a[href*='/products/']", 
                    ".product-list a",
                    ".product-card a",
                    ".card a",
                    "a.product-link",
                    "#product-list a"
                ]
                
                product_elements = []
                for selector in product_selectors:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        product_elements.extend(elements)
                        logger.info(f"Found {len(elements)} products with selector {selector}")
                
                for element in product_elements:
                    try:
                        href = element.get_attribute("href")
                        text = element.text.strip()
                        
                        # If no text was found, try to get it from the parent
                        if not text:
                            try:
                                parent = element.find_element(By.XPATH, "..")
                                if parent:
                                    text = parent.text.strip()
                            except:
                                pass
                        
                        if href and text and ("view" in href or "products" in href or "catalog" in href):
                            assessment_links.append({
                                "url": href,
                                "name": text
                            })
                    except Exception as inner_e:
                        logger.warning(f"Error processing element: {inner_e}")
                        continue
                
                # Add some randomized wait time to avoid being detected as a bot
                time.sleep(random.uniform(3, 5))
                
            except Exception as e:
                logger.error(f"Error getting assessment links from {url}: {e}")
        
        # Remove duplicates based on URL
        unique_links = {}
        for item in assessment_links:
            if item["url"] not in unique_links and item["name"]:
                unique_links[item["url"]] = item["name"]
        
        result = [{"url": url, "name": name} for url, name in unique_links.items()]
        logger.info(f"Found {len(result)} unique assessment links")
        
        return result
    
    def _extract_metadata_selenium(self, field: str) -> str:
        """
        Extract assessment metadata using Selenium.
        
        Args:
            field: The metadata field to extract (duration, remote, adaptive, test_type)
            
        Returns:
            Metadata value as string
        """
        try:
            # Define field-specific keywords and patterns
            field_keywords = {
                "duration": ["duration", "test time", "assessment time"],
                "remote": ["remote", "remote testing", "online"],
                "adaptive": ["adaptive", "adaptive testing"],
                "test_type": ["test type", "assessment type", "type"]
            }
            
            # First try product attributes table
            try:
                product_attrs = self.driver.find_elements(By.CSS_SELECTOR, ".product-attributes tr")
                for attr_row in product_attrs:
                    try:
                        cells = attr_row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 2:
                            attr_name = cells[0].text.strip().lower()
                            attr_value = cells[1].text.strip()
                            
                            for keyword in field_keywords[field]:
                                if keyword in attr_name:
                                    return attr_value
                    except:
                        continue
            except:
                pass
            
            # Try to find labeled elements
            keywords = field_keywords[field]
            for keyword in keywords:
                try:
                    xpath_expressions = [
                        f"//strong[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword}')]/following-sibling::text()",
                        f"//strong[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword}')]/parent::*/following-sibling::*",
                        f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword}:')]/following-sibling::*",
                        f"//*[contains(@class, '{keyword}')]"
                    ]
                    
                    for xpath in xpath_expressions:
                        elements = self.driver.find_elements(By.XPATH, xpath)
                        if elements:
                            value = elements[0].text.strip()
                            if value:
                                return value
                except:
                    continue
            
            # For duration, try to extract from description
            if field == "duration":
                try:
                    description = self._extract_description_selenium()
                    if description:
                        duration_patterns = [
                            r'(?:duration|test time|assessment time)[:\s]+([\d-]+)\s*(?:min|minutes)',
                            r'([\d-]+)\s*(?:min|minutes)\s*(?:duration|test|assessment)',
                            r'takes\s+([\d-]+)\s*(?:min|minutes)'
                        ]
                        
                        for pattern in duration_patterns:
                            match = re.search(pattern, description.lower())
                            if match:
                                return f"{match.group(1)} minutes"
                except:
                    pass
            
            # For test_type, try to extract from description
            if field == "test_type" and self._extract_description_selenium():
                description = self._extract_description_selenium().lower()
                test_types = [
                    "cognitive", "personality", "behavioral", "situational judgment",
                    "coding", "technical", "numerical", "verbal", "abstract", "logical"
                ]
                
                for t_type in test_types:
                    if t_type in description:
                        return t_type.title()
            
            return ""
        except Exception as e:
            logger.error(f"Error extracting {field}: {e}")
            return ""
    
    def _clean_field(self, value: str) -> str:
        """
        Clean a field value.
        
        Args:
            value: The field value to clean
            
        Returns:
            Cleaned field value
        """
        if not value:
            return ""
            
        value = str(value).strip()
        
        # Remove cookie consent messages
        if ("cookie" in value.lower() or 
            "permission" in value.lower() or 
            "consent" in value.lower()):
            return ""
            
        return value
    
    def scrape_assessments(self) -> pd.DataFrame:
        """
        Scrape assessment details from SHL website.
        
        Returns:
            DataFrame containing assessment details
        """
        logger.info("Starting assessment scraping process")
        
        # Visit the main page and handle cookie consent
        try:
            self.driver.get(self.base_url)
            self._handle_cookie_consent()
        except Exception as e:
            logger.error(f"Error visiting main page: {e}")
        
        # Get assessment links more effectively
        assessment_links = self._get_assessment_links()
        assessment_data = []
        
        for i, link_info in enumerate(assessment_links):
            try:
                link = link_info["url"]
                name = link_info["name"]
                
                logger.info(f"Processing link {i+1}/{len(assessment_links)}: {name} ({link})")
                
                # Navigate to the assessment page
                self.driver.get(link)
                
                # Wait for page to load with multiple possible content selectors
                page_loaded = False
                content_selectors = [".content", ".main-content", ".product-detail", ".product-page", "body"]
                
                for selector in content_selectors:
                    try:
                        WebDriverWait(self.driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        page_loaded = True
                        logger.info(f"Page loaded, found selector: {selector}")
                        break
                    except TimeoutException:
                        continue
                
                if not page_loaded:
                    logger.warning(f"Timeout waiting for content on {link}")
                    # Continue anyway and try to extract what we can
                
                # Add a small delay for JavaScript content
                time.sleep(random.uniform(1, 2))
                
                # Extract assessment details directly from the page
                description = self._extract_description_selenium()
                duration = self._extract_metadata_selenium("duration")
                remote = self._extract_metadata_selenium("remote")
                adaptive = self._extract_metadata_selenium("adaptive")
                test_type = self._extract_metadata_selenium("test_type")
                
                # Clean the data
                description = self._clean_field(description)
                duration = self._clean_field(duration)
                remote = self._clean_field(remote)
                adaptive = self._clean_field(adaptive)
                test_type = self._clean_field(test_type)
                
                # Add to assessment data
                assessment_data.append({
                    "name": name,
                    "url": link,
                    "description": description if description else "Not specified",
                    "duration": duration if duration else "Not specified",
                    "remote": remote if remote else "Not specified",
                    "adaptive": adaptive if adaptive else "Not specified",
                    "test_type": test_type if test_type else "Not specified"
                })
                
                # Random wait between requests to be polite
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error processing link {link}: {e}")
                # Still add the basic info we have
                assessment_data.append({
                    "name": name if 'name' in locals() else "Unknown",
                    "url": link if 'link' in locals() else "",
                    "description": "Not specified",
                    "duration": "Not specified",
                    "remote": "Not specified",
                    "adaptive": "Not specified",
                    "test_type": "Not specified"
                })
        
        # Create DataFrame
        df = pd.DataFrame(assessment_data)
        logger.info(f"Scraped {len(df)} assessments")
        
        return df
    
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
