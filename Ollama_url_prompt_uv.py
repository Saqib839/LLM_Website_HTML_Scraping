"""
Doctor Data Scraper using Ollama LLM
Reads website URLs from req_input.csv and scrapes doctor information
as per the requirements in Doctor part.docx

This version uses Ollama LLM for intelligent extraction of doctor information
instead of rule-based pattern matching. This provides more accurate and
contextual extraction of doctor data from websites.

Requirements:
1. Install Ollama from https://ollama.ai/
2. Pull a model: ollama pull llama3 (or another model)
3. Ensure Ollama is running (default: http://localhost:11434)

Environment Variables (optional):
- OLLAMA_MODEL: Model name to use (default: 'llama3')
- OLLAMA_BASE_URL: Ollama API base URL (default: 'http://localhost:11434')
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time
import random
from typing import List, Dict, Optional
import logging
import os
import json
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DoctorScraper:
    """
    Scrapes doctor information from practice websites using Ollama LLM
    Per requirements: Extract full name, bio, age, hometown, education, photo
    """
    
    def __init__(self, ollama_model: str = 'llama3', ollama_base_url: str = 'http://localhost:11434'):
        """
        Initialize the scraper with session, headers, and Ollama configuration
        
        Args:
            ollama_model: Name of the Ollama model to use (default: 'llama3')
            ollama_base_url: Base URL for Ollama API (default: 'http://localhost:11434')
        """
        self.current_year = datetime.now().year
        self.session = requests.Session()
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip('/')
        
        # Rotating User Agents to avoid detection
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        
        self.timeout = 20  # Reduced from 30
        self.base_delay = 0.3  # Reduced from 2 - minimal delay
        self.max_delay = 0.8   # Reduced from 5 - minimal delay
        
        # Test Ollama connection
        self._test_ollama_connection()
    
    def read_input_csv(self, csv_file: str = 'req_input.csv') -> List[str]:
        """
        Read website URLs from CSV file
        Supports req_input.csv or Req_Inpu.csv (typo variant)
        
        Args:
            csv_file: Path to CSV file containing URLs
            
        Returns:
            List of website URLs
        """
        urls = []
        
        # Try different possible filenames
        possible_files = [csv_file, 'req_input.csv', 'Req_Inpu.csv', 'Req_Input.csv']
        file_path = None
        
        for filename in possible_files:
            if os.path.exists(filename):
                file_path = filename
                logger.info(f"Found input file: {filename}")
                break
        
        if not file_path:
            logger.error(f"Input CSV file not found. Tried: {possible_files}")
            return urls
        
        try:
            # Try reading as CSV with pandas first
            df = pd.read_csv(file_path, header=None)
            
            # Extract URLs from all columns
            for column in df.columns:
                for value in df[column].dropna():
                    url = str(value).strip()
                    if url and url.startswith('http'):
                        urls.append(url)
            
            # If no URLs found, try reading line by line
            if not urls:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and (line.startswith('http') or 'www.' in line or '.com' in line):
                            if not line.startswith('http'):
                                line = 'https://' + line
                            urls.append(line)
            
            logger.info(f"Read {len(urls)} URLs from {file_path}")
            return urls
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return urls
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Ollama connection successful. Using model: {self.ollama_model}")
            else:
                logger.warning(f"Ollama connection test returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {self.ollama_base_url}: {e}")
            logger.warning("Please ensure Ollama is running. You can install it from https://ollama.ai/")
    
    def _call_ollama(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call Ollama API to generate a response
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retry attempts
            
        Returns:
            LLM response text or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        'model': self.ollama_model,
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': 0.1,  # Low temperature for more deterministic extraction
                            'num_predict': 3000  # Increased to handle more doctors per page
                        }
                    },
                    timeout=60  # Increased from 45 - give LLM more time to extract all doctors
                )
                response.raise_for_status()
                result = response.json()
                return result.get('response', '')
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama request timeout, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(1)
                else:
                    logger.warning(f"Ollama request timeout after {max_retries} attempts, will use fallback extraction")
                    return None
            except Exception as e:
                logger.error(f"Error calling Ollama: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None
        return None
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response, handling various formats
        Supports both arrays and objects
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON (list, dict, or None)
        """
        if not response:
            return None
        
        # Clean up response - remove leading/trailing whitespace
        response = response.strip()
        
        # Try to find JSON in markdown code blocks (array or object)
        json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array first (most common format we expect)
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try parsing entire response as JSON (could be array or object)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Last resort: try to extract and fix common JSON issues
        # Remove any text before first [ or {
        cleaned = re.sub(r'^[^\[\{]*', '', response)
        # Remove any text after last ] or }
        cleaned = re.sub(r'[^\]\}]*$', '', cleaned)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def extract_doctors_with_llm(self, html_content: str, base_url: str) -> List[Dict]:
        """
        Extract doctor information using Ollama LLM
        
        Args:
            html_content: HTML content or text from the webpage
            base_url: Base URL of the website
            
        Returns:
            List of doctor dictionaries
        """
        # Fast HTML parsing - use lxml if available, otherwise html.parser
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and other non-content elements (faster than decompose)
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
            tag.decompose()
        
        # Get clean text content (faster method)
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Limit text length to avoid token limits (keep first 12000 chars, prioritize doctor-related sections)
        if len(text_content) > 12000:
            # Fast keyword-based filtering
            doctor_keywords = ['doctor', 'dentist', 'dds', 'dmd', 'meet', 'team', 'staff', 'orthodontist', 'dr.', 'dr ']
            words = text_content.split()
            important_words = []
            other_words = []
            
            for word in words:
                word_lower = word.lower()
                if any(keyword in word_lower for keyword in doctor_keywords):
                    important_words.append(word)
                else:
                    other_words.append(word)
            
            # Combine: all important words + some other words
            if len(important_words) > 0:
                text_content = ' '.join(important_words) + ' ' + ' '.join(other_words[:2000])
                if len(text_content) > 12000:
                    text_content = text_content[:12000]
            else:
                text_content = text_content[:12000]
        
        # Create prompt for LLM with more explicit JSON format instructions
        prompt = f"""Extract ALL doctor information from this dental practice website. Return ONLY a valid JSON array, no other text.

CRITICAL: You MUST find ALL doctors mentioned, even if they only have a name and no other details. Include doctors with minimal information.

REQUIREMENTS:
- Find ALL doctors mentioned (orthodontists, pediatric dentists, general dentists, periodontists, hygienists with names, etc.)
- Extract full names (remove "Dr.", "Doctor", "DDS", "DMD", "MS", "R.D.H." but keep the actual name)
- Extract bio text if available (can be empty if not found)
- Calculate age from graduation year (assume graduation at age 26, current year is {self.current_year})
- Extract hometown if mentioned
- Extract education/dental school information
- Extract photo URLs if available
- Include doctors even if they only appear as names in lists without detailed bios

OUTPUT FORMAT - Return ONLY this JSON structure (no markdown, no explanations):
[
  {{
    "name": "Full Name Here",
    "bio": "Biography text here or empty string",
    "age": 45 or null,
    "hometown": "City, State" or "",
    "education": "School name, year" or "",
    "photo_url": "https://..." or ""
  }}
]

IMPORTANT: 
- Return ONLY the JSON array, nothing else
- Use double quotes for JSON
- Include ALL doctors found, even if some fields are empty (name is required, other fields can be empty strings or null)
- Look carefully through the entire page - doctors may be listed in sections, cards, or lists
- If no doctors found, return empty array: []
- Count ALL unique doctor names, even if bio is missing

Website content:
{text_content}

JSON array:"""

        llm_response = self._call_ollama(prompt)
        
        if not llm_response:
            # Use fallback immediately - faster than retrying
            return self._fallback_extraction(text_content, base_url)
        
        # Parse JSON response
        doctors = self._extract_json_from_response(llm_response)
        
        if not doctors:
            logger.debug("Could not parse LLM response as JSON, using fallback extraction")
            # Try to extract manually from text response (faster than retrying LLM)
            return self._fallback_extraction(text_content, base_url)
        
        # Ensure it's a list
        if isinstance(doctors, dict):
            doctors = [doctors]
        elif not isinstance(doctors, list):
            logger.warning(f"Unexpected response type: {type(doctors)}, trying fallback")
            return self._fallback_extraction(text_content, base_url)
        
        # Validate and clean doctor data
        validated_doctors = []
        for doctor in doctors:
            if isinstance(doctor, dict) and doctor.get('name'):
                # Convert photo URLs to absolute URLs
                photo_url = doctor.get('photo_url', '')
                if photo_url and not photo_url.startswith('http'):
                    if photo_url.startswith('/'):
                        doctor['photo_url'] = urljoin(base_url, photo_url)
                    else:
                        doctor['photo_url'] = urljoin(base_url + '/', photo_url)
                
                # Clean bio before adding
                bio = self._clean_bio(str(doctor.get('bio', '')).strip())
                
                validated_doctors.append({
                    'name': str(doctor.get('name', '')).strip(),
                    'bio': bio,
                    'age': doctor.get('age') if isinstance(doctor.get('age'), int) else None,
                    'hometown': str(doctor.get('hometown', '')).strip(),
                    'education': str(doctor.get('education', '')).strip(),
                    'photo_url': doctor.get('photo_url', '')
                })
        
        if len(validated_doctors) > 0:
            logger.info(f"LLM extracted {len(validated_doctors)} doctors")
        return validated_doctors
    
    def _fallback_extraction(self, text_content: str, base_url: str) -> List[Dict]:
        """
        Fallback extraction method using regex patterns if LLM fails
        Improved to handle various name formats
        
        Args:
            text_content: Text content from webpage
            base_url: Base URL of the website
            
        Returns:
            List of doctor dictionaries
        """
        doctors = []
        seen_names = set()
        
        # Pattern 1: "Dr. FirstName LastName" or "Dr FirstName LastName"
        doctor_pattern1 = r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)'
        matches = re.finditer(doctor_pattern1, text_content)
        for match in matches:
            name = match.group(1).strip()
            # Remove common suffixes
            name = re.sub(r'\s+(DDS|DMD|MS|D\.D\.S\.|D\.M\.D\.)$', '', name, flags=re.I)
            name_key = name.lower()
            if name_key and name_key not in seen_names and len(name.split()) >= 2:
                seen_names.add(name_key)
                doctors.append({
                    'name': name,
                    'bio': '',
                    'age': None,
                    'hometown': '',
                    'education': '',
                    'photo_url': ''
                })
        
        # Pattern 2: "FirstName LastName, DDS" or "FirstName LastName DDS"
        doctor_pattern2 = r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[,]?\s*(?:DDS|DMD|MS|D\.D\.S\.|D\.M\.D\.)'
        matches = re.finditer(doctor_pattern2, text_content)
        for match in matches:
            name = match.group(1).strip()
            name_key = name.lower()
            if name_key and name_key not in seen_names and len(name.split()) >= 2:
                seen_names.add(name_key)
                doctors.append({
                    'name': name,
                    'bio': '',
                    'age': None,
                    'hometown': '',
                    'education': '',
                    'photo_url': ''
                })
        
        # Pattern 3: Look for sections with "Meet the Doctors" or similar headers
        # Extract names from structured sections
        section_patterns = [
            r'(?:Meet\s+the\s+Doctors?|Our\s+Doctors?|Pediatric\s+Dentists?|Orthodontists?|General\s+Dentists?)[:\s]*\n(.*?)(?:\n\n|\n[A-Z][a-z]+\s+[A-Z]|$)',
        ]
        
        for pattern in section_patterns:
            sections = re.finditer(pattern, text_content, re.IGNORECASE | re.DOTALL)
            for section in sections:
                section_text = section.group(1) if section.groups() else section.group(0)
                # Extract names from this section
                name_matches = re.finditer(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', section_text)
                for name_match in name_matches:
                    name = name_match.group(1).strip()
                    name_key = name.lower()
                    # Filter out common false positives
                    if (name_key not in seen_names and 
                        len(name.split()) >= 2 and 
                        name_key not in ['meet the', 'our team', 'pediatric dental', 'general dentist']):
                        seen_names.add(name_key)
                        doctors.append({
                            'name': name,
                            'bio': '',
                            'age': None,
                            'hometown': '',
                            'education': '',
                            'photo_url': ''
                        })
        
        if len(doctors) > 0:
            logger.info(f"Fallback extraction found {len(doctors)} doctors")
        return doctors
    
    def get_random_headers(self, referer: Optional[str] = None) -> Dict[str, str]:
        """
        Generate random browser-like headers to avoid detection
        
        Args:
            referer: Optional referer URL
            
        Returns:
            Dictionary of headers
        """
        user_agent = random.choice(self.user_agents)
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none' if not referer else 'same-origin',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
        }
        
        if referer:
            headers['Referer'] = referer
        
        return headers
    
    def random_delay(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None):
        """
        Add random delay to mimic human behavior
        
        Args:
            min_seconds: Minimum delay in seconds
            max_seconds: Maximum delay in seconds
        """
        if min_seconds is None:
            min_seconds = self.base_delay
        if max_seconds is None:
            max_seconds = self.max_delay
        
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def establish_session(self, base_url: str) -> bool:
        """
        Establish a session by visiting the homepage first
        This helps avoid bot detection
        
        Args:
            base_url: Base URL of the website
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Establishing session with {base_url}")
            headers = self.get_random_headers()
            response = self.session.get(
                base_url, 
                headers=headers,
                timeout=self.timeout, 
                allow_redirects=True
            )
            
            # Add cookies from response
            self.session.cookies.update(response.cookies)
            
            # Minimal delay
            time.sleep(0.2)
            
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Error establishing session: {e}")
            return False
    
    def scrape_website(self, url: str, retries: int = 2) -> Optional[BeautifulSoup]:
        """
        Scrape website and return BeautifulSoup object with enhanced anti-bot detection
        
        Args:
            url: Website URL to scrape
            retries: Number of retry attempts
            
        Returns:
            BeautifulSoup object or None if scraping fails
        """
        if not url or not url.strip():
            logger.warning(f"Invalid URL: {url}")
            return None
        
        # Normalize URL
        url = url.strip()
        if not url.startswith('http'):
            url = 'https://' + url
        
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Establish session by visiting homepage first (helps avoid detection)
        session_established = False
        
        for attempt in range(retries + 1):
            try:
                # Rotate user agent and headers for each attempt
                referer = base_url if session_established else None
                headers = self.get_random_headers(referer=referer)
                
                # Update session headers
                self.session.headers.update(headers)
                
                # Skip session establishment for speed - only do it if we get 403
                
                logger.debug(f"Attempting to scrape {url} (attempt {attempt + 1}/{retries + 1})")
                
                # Make the request
                response = self.session.get(
                    url, 
                    headers=headers,
                    timeout=self.timeout, 
                    allow_redirects=True
                )
                
                # If we get 403, try different strategies
                if response.status_code == 403:
                    if attempt == 0:
                        # Strategy 1: Try establishing session first
                        logger.info(f"Got 403, trying to establish session first...")
                        if self.establish_session(base_url):
                            session_established = True
                            # Minimal wait
                            time.sleep(1)
                            # Retry with new session
                            continue
                    
                    # Strategy 2: Try with different user agent
                    if attempt < retries:
                        wait_time = (attempt + 1) * random.uniform(3, 6)
                        logger.warning(f"403 Forbidden - Waiting {wait_time:.1f} seconds before retry with different headers...")
                        time.sleep(wait_time)
                        
                        # Rotate to a completely different user agent
                        headers = self.get_random_headers(referer=base_url)
                        self.session.headers.clear()
                        self.session.headers.update(headers)
                        continue
                    else:
                        logger.error(f"Website blocked access after {retries + 1} attempts: {url}")
                        return None
                
                response.raise_for_status()
                
                # Success - parse the HTML
                soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
                logger.info(f"Successfully scraped {url}")
                
                # Minimal delay after successful request
                time.sleep(0.1)
                
                return soup
                
            except requests.exceptions.Timeout as e:
                if attempt < retries:
                    wait_time = (attempt + 1) * random.uniform(3, 6)
                    logger.warning(f"Timeout scraping {url}, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Timeout scraping website {url} after {retries + 1} attempts: {e}")
                    
            except requests.exceptions.HTTPError as e:
                if attempt < retries:
                    wait_time = (attempt + 1) * random.uniform(3, 6)
                    logger.warning(f"HTTP error scraping {url}, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error scraping website {url}: {e}")
                        
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    wait_time = (attempt + 1) * random.uniform(2, 4)
                    logger.warning(f"Error scraping {url}, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error scraping website {url}: {e}")
                    
            except Exception as e:
                logger.error(f"Unexpected error scraping {url}: {e}")
                return None
            
            # Minimal delay between retry attempts
            if attempt < retries:
                time.sleep(0.5)
        
        return None
    
    def find_doctor_pages(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find URLs that likely contain doctor information
        
        Args:
            soup: BeautifulSoup object of the main page
            base_url: Base URL of the website
            
        Returns:
            List of URLs that likely contain doctor information
        """
        doctor_urls = set()
        
        # Prioritized keywords (most likely to contain doctor info)
        priority_keywords = ['doctor', 'dentist', 'meet', 'team', 'staff', 'bio', 'biography']
        other_keywords = ['physician', 'about', 'provider', 'our']
        
        parsed_base = urlparse(base_url)
        
        # Limit search to first 100 links for speed
        links = soup.find_all('a', href=True, limit=100)
        
        for link in links:
            href = link.get('href', '')
            if not href:
                continue
                
            href_lower = href.lower()
            text_lower = link.get_text().lower().strip()
            
            # Check priority keywords first
            has_priority = any(keyword in href_lower or keyword in text_lower for keyword in priority_keywords)
            has_other = any(keyword in href_lower or keyword in text_lower for keyword in other_keywords)
            
            if has_priority or has_other:
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                # Only add if it's from the same domain
                if parsed_url.netloc == parsed_base.netloc or not parsed_url.netloc:
                    doctor_urls.add(full_url)
                    # Stop early if we have enough URLs (increased to catch more pages)
                    if len(doctor_urls) >= 8:
                        break
        
        return list(doctor_urls)
    
    def find_individual_doctor_pages(self, soup: BeautifulSoup, base_url: str, doctors: List[Dict]) -> Dict[str, str]:
        """
        Find individual doctor profile pages by matching doctor names with links
        
        Args:
            soup: BeautifulSoup object of the webpage
            base_url: Base URL of the website
            doctors: List of doctors already found (to match names)
            
        Returns:
            Dictionary mapping doctor profile URLs to doctor names
        """
        individual_pages = {}
        parsed_base = urlparse(base_url)
        
        # Get all links from the page
        links = soup.find_all('a', href=True)
        
        # Extract doctor names (first and last name parts)
        doctor_name_parts = []
        for doctor in doctors:
            name = doctor.get('name', '').strip()
            if name:
                parts = name.lower().split()
                if len(parts) >= 2:
                    doctor_name_parts.append({
                        'full': name.lower(),
                        'first': parts[0],
                        'last': parts[-1],
                        'original': name
                    })
        
        # Also look for doctor names in text and find associated links
        for link in links:
            href = link.get('href', '')
            if not href:
                continue
                
            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)
            
            # Only check same domain links
            if parsed_url.netloc != parsed_base.netloc and parsed_url.netloc:
                continue
            
            href_lower = href.lower()
            link_text = link.get_text().lower().strip()
            
            # Check if link might be to an individual doctor page
            # Look for patterns like /doctor/name, /dentist/name, /about/name, etc.
            individual_patterns = [
                r'/(?:doctor|dentist|physician|about|team|staff)/[^/]+',
                r'/dr[-_]?[^/]+',
                r'/[a-z-]+[-_](?:dds|dmd)',
            ]
            
            is_individual_page = False
            matched_name = None
            
            # Check if URL matches individual doctor page patterns
            for pattern in individual_patterns:
                if re.search(pattern, href_lower):
                    is_individual_page = True
                    break
            
            # Also check if link text contains a doctor name
            for name_info in doctor_name_parts:
                if (name_info['first'] in link_text or name_info['last'] in link_text or
                    name_info['first'] in href_lower or name_info['last'] in href_lower):
                    is_individual_page = True
                    matched_name = name_info['original']
                    break
            
            # Check if link is near doctor names in the HTML
            if not is_individual_page and doctor_name_parts:
                # Get surrounding context
                parent = link.parent
                if parent:
                    context_text = parent.get_text().lower()
                    for name_info in doctor_name_parts:
                        # Check if doctor name appears near this link
                        if name_info['first'] in context_text and name_info['last'] in context_text:
                            # Calculate distance
                            link_idx = context_text.find(link_text)
                            name_idx = context_text.find(name_info['full'])
                            if link_idx >= 0 and name_idx >= 0 and abs(link_idx - name_idx) < 200:
                                is_individual_page = True
                                matched_name = name_info['original']
                                break
            
            if is_individual_page and full_url not in individual_pages:
                if matched_name:
                    individual_pages[full_url] = matched_name
                else:
                    # Try to extract name from URL or link text
                    name_from_url = self._extract_name_from_url(href, link_text)
                    if name_from_url:
                        individual_pages[full_url] = name_from_url
        
        return individual_pages
    
    def _extract_name_from_url(self, href: str, link_text: str) -> Optional[str]:
        """Extract doctor name from URL or link text"""
        # Try to extract name from link text first
        name_match = re.search(r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', link_text, re.I)
        if name_match:
            return name_match.group(1).strip()
        
        # Try to extract from URL
        url_parts = href.split('/')
        for part in reversed(url_parts):
            if part and len(part) > 3:
                # Replace hyphens/underscores with spaces and capitalize
                name = part.replace('-', ' ').replace('_', ' ')
                words = name.split()
                if len(words) >= 2:
                    capitalized = ' '.join([w.capitalize() for w in words])
                    return capitalized
        
        return None
    
    def extract_detailed_doctor_info(self, html_content: str, base_url: str, doctor_name: str) -> Dict:
        """
        Extract detailed information from an individual doctor's profile page
        
        Args:
            html_content: HTML content of the doctor's profile page
            base_url: Base URL of the page
            doctor_name: Name of the doctor (for verification)
            
        Returns:
            Dictionary with detailed doctor information
        """
        doctor = {
            'name': doctor_name,
            'bio': '',
            'age': None,
            'hometown': '',
            'education': '',
            'photo_url': ''
        }
        
        # Use LLM to extract detailed info from the profile page
        detailed_doctors = self.extract_doctors_with_llm(html_content, base_url)
        
        # Find the matching doctor in the extracted list
        for doc in detailed_doctors:
            extracted_name = doc.get('name', '').lower().strip()
            target_name = doctor_name.lower().strip()
            
            # Check if names match (allowing for variations)
            if (extracted_name == target_name or 
                extracted_name.replace('.', '') == target_name.replace('.', '') or
                self._names_match(extracted_name, target_name)):
                
                # Use the detailed information (clean bio)
                doctor.update(doc)
                if doctor.get('bio'):
                    doctor['bio'] = self._clean_bio(doctor['bio'])
                break
        
        # If LLM didn't find detailed info, try manual extraction
        if not doctor.get('bio') or len(doctor.get('bio', '')) < 50:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for bio sections
            bio_selectors = [
                {'class': re.compile(r'bio|about|description|text|content', re.I)},
                {'id': re.compile(r'bio|about|description', re.I)},
            ]
            
            for selector in bio_selectors:
                bio_elements = soup.find_all(['div', 'section', 'article', 'p'], selector)
                if bio_elements:
                    bio_text = ' '.join([elem.get_text(strip=True) for elem in bio_elements])
                    if len(bio_text) > len(doctor.get('bio', '')):
                        doctor['bio'] = self._clean_bio(bio_text[:2000])
            
            # Extract photo if not found
            if not doctor.get('photo_url'):
                img_tags = soup.find_all('img', src=True)
                for img in img_tags:
                    img_src = img.get('src', '')
                    img_alt = img.get('alt', '').lower()
                    if doctor_name.split()[0].lower() in img_alt or 'doctor' in img_alt or 'dentist' in img_alt:
                        if img_src.startswith('/'):
                            doctor['photo_url'] = urljoin(base_url, img_src)
                        elif img_src.startswith('http'):
                            doctor['photo_url'] = img_src
                        break
        
        return doctor
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names match (allowing for variations)"""
        name1_parts = set(name1.split())
        name2_parts = set(name2.split())
        
        # If they share at least 2 words, likely a match
        if len(name1_parts & name2_parts) >= 2:
            return True
        
        # Check if last names match and first names start with same letter
        name1_list = name1.split()
        name2_list = name2.split()
        if len(name1_list) >= 2 and len(name2_list) >= 2:
            if name1_list[-1] == name2_list[-1] and name1_list[0][0] == name2_list[0][0]:
                return True
        
        return False
    
    def extract_doctors_from_website(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """
        Extract doctor information from website using LLM
        Per requirements: Extract full name, bio, age, hometown, education, photo
        
        Args:
            soup: BeautifulSoup object of the webpage
            base_url: Base URL of the website
            
        Returns:
            List of doctor dictionaries with all required fields
        """
        doctors = []
        
        if not soup:
            return doctors
        
        # Convert soup to HTML string for LLM processing
        html_content = str(soup)
        
        # Primary extraction using LLM from main page
        main_page_doctors = self.extract_doctors_with_llm(html_content, base_url)
        
        # Deduplicate by name
        seen_names = {doc.get('name', '').lower().strip() for doc in main_page_doctors}
        doctors.extend(main_page_doctors)
        
        # Always find and visit doctor pages to get complete information
        doctor_urls = self.find_doctor_pages(soup, base_url)
        logger.info(f"Found {len(doctor_urls)} potential doctor page URLs")
        
        # Step 1: Visit team/about pages to find more doctors
        team_pages = [url for url in doctor_urls if any(kw in url.lower() for kw in ['team', 'about', 'meet', 'staff', 'doctors'])]
        for url in team_pages[:5]:  # Visit up to 5 team pages
            try:
                doctor_soup = self.scrape_website(url)
                if doctor_soup:
                    doctor_html = str(doctor_soup)
                    page_doctors = self.extract_doctors_with_llm(doctor_html, url)
                    
                    # Add new doctors not already seen (clean bio first)
                    for doc in page_doctors:
                        name_key = doc.get('name', '').lower().strip()
                        if name_key and name_key not in seen_names:
                            if doc.get('bio'):
                                doc['bio'] = self._clean_bio(doc['bio'])
                            seen_names.add(name_key)
                            doctors.append(doc)
                    
                    time.sleep(0.2)
            except Exception as e:
                logger.debug(f"Error processing team page {url}: {e}")
                continue
        
        # Step 2: Find and visit individual doctor profile pages for detailed bios
        # First, check the main page
        individual_doctor_urls = self.find_individual_doctor_pages(soup, base_url, doctors)
        
        # Also check team pages for individual doctor links
        for team_url in team_pages[:3]:  # Check first 3 team pages
            try:
                team_soup = self.scrape_website(team_url)
                if team_soup:
                    team_individual_urls = self.find_individual_doctor_pages(team_soup, team_url, doctors)
                    individual_doctor_urls.update(team_individual_urls)
            except:
                pass
        
        logger.info(f"Found {len(individual_doctor_urls)} individual doctor profile pages")
        
        # Visit each doctor's individual page to get detailed bio
        for doctor_url, doctor_name in individual_doctor_urls.items():
            try:
                doctor_soup = self.scrape_website(doctor_url)
                if doctor_soup:
                    doctor_html = str(doctor_soup)
                    detailed_info = self.extract_detailed_doctor_info(doctor_html, doctor_url, doctor_name)
                    
                    # Update existing doctor entry or add new one
                    doctor_found = False
                    for i, doc in enumerate(doctors):
                        if doc.get('name', '').lower().strip() == doctor_name.lower().strip():
                            # Update with detailed info (clean bio first)
                            if detailed_info.get('bio'):
                                cleaned_bio = self._clean_bio(detailed_info['bio'])
                                if len(cleaned_bio) > len(doc.get('bio', '')):
                                    doc['bio'] = cleaned_bio
                            if detailed_info.get('education') and not doc.get('education'):
                                doc['education'] = detailed_info.get('education', '')
                            if detailed_info.get('hometown') and not doc.get('hometown'):
                                doc['hometown'] = detailed_info.get('hometown', '')
                            if detailed_info.get('age') and not doc.get('age'):
                                doc['age'] = detailed_info.get('age')
                            if detailed_info.get('photo_url') and not doc.get('photo_url'):
                                doc['photo_url'] = detailed_info.get('photo_url', '')
                            doctor_found = True
                            break
                    
                    # If doctor not in list yet, add them (clean bio first)
                    if not doctor_found and detailed_info.get('name'):
                        if detailed_info.get('bio'):
                            detailed_info['bio'] = self._clean_bio(detailed_info['bio'])
                        doctors.append(detailed_info)
                        seen_names.add(detailed_info.get('name', '').lower().strip())
                    
                    time.sleep(0.2)
            except Exception as e:
                logger.debug(f"Error processing doctor profile page {doctor_url}: {e}")
                continue
        
        # Clean bio fields to remove unnecessary content
        for doctor in doctors:
            if doctor.get('bio'):
                doctor['bio'] = self._clean_bio(doctor['bio'])
        
        logger.info(f"Extracted {len(doctors)} doctors from website")
        return doctors
    
    def _clean_bio(self, bio: str) -> str:
        """
        Clean bio text to remove navigation, footer, form messages, and other unnecessary content
        
        Args:
            bio: Raw bio text
            
        Returns:
            Cleaned bio text
        """
        if not bio:
            return ''
        
        # Remove common navigation/footer patterns (compiled for speed)
        unwanted_patterns = [
            r'CONTACT US.*?BLOG.*?All Rights Reserved.*',
            r'Search \.\.\. Search.*',
            r'Pay Online.*?Patient Portal.*?News.*?Staff.*',
            r'Thank you!.*?submission.*?received.*',
            r'Oops!.*?Something went wrong.*',
            r'Monday.*?Sunday.*?Closed.*',
            r'Phone:.*?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}.*',
            r'Address:.*?,\s*\w+\s+\d{5}.*',
            r'Email:.*?@.*',
            r'BLOG.*?CONTACT.*?BLOG.*',
            r'All Rights Reserved.*?Maintained by.*',
            r'Home.*?About.*?Services.*?Contact.*',
            r'Back.*?Back.*?Back.*',  # Navigation breadcrumbs
            r'Provider Search.*?Find the Provider.*',
            r'Schedule.*?Appointment.*?Request.*',
            r'My Account.*?Pay My Bill.*',
            r'©\s*\d{4}.*?All rights reserved.*',
            r'Website by.*',
            r'Maintained by.*',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}.*',  # Phone numbers anywhere
            r'\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Circle|Ct)\.?.*',  # Addresses
            r'Appointment.*?Request.*',
            r'schedule now.*',
            r'CONTACT US.*?\d{3}.*?\d{4}.*',  # Contact info with phone
            r'Fort Lauderdale.*?Florida.*?\d{5}.*',  # Specific address patterns
            r'\d{1,2}:\d{2}\s*(?:AM|PM).*?\d{1,2}:\d{2}\s*(?:AM|PM).*',  # Hours
        ]
        
        cleaned = bio
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove repeated phrases (optimized for speed - only check if bio is long)
        if len(cleaned) > 500:
            # Simple approach: remove obvious repeated sections
            # Split by common separators and remove duplicates
            sentences = re.split(r'[.!?]\s+', cleaned)
            seen_sentences = set()
            unique_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.strip().lower()[:50]  # First 50 chars for comparison
                if sentence_lower and sentence_lower not in seen_sentences:
                    seen_sentences.add(sentence_lower)
                    unique_sentences.append(sentence.strip())
            cleaned = '. '.join(unique_sentences)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        # Remove lines that are just navigation items (optimized)
        if '\n' in cleaned:
            lines = cleaned.split('\n')
            nav_keywords = {'home', 'about', 'services', 'contact', 'blog', 'back', 'next', 'previous', 'menu', 'navigation'}
            filtered_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped or len(line_stripped) < 3:
                    continue
                # Skip short navigation items
                if len(line_stripped) < 30:
                    line_lower = line_stripped.lower()
                    if any(keyword in line_lower for keyword in nav_keywords):
                        continue
                    # Skip phone numbers or hours
                    if re.match(r'^[\d\s\-\(\)]+$', line_stripped) or ('AM' in line_stripped and 'PM' in line_stripped):
                        continue
                filtered_lines.append(line)
            
            cleaned = '\n'.join(filtered_lines)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        # Limit length to reasonable size (2000 chars)
        if len(cleaned) > 2000:
            cleaned = cleaned[:2000] + '...'
        
        return cleaned
    
    def find_doctor_sections_on_page(self, soup: BeautifulSoup) -> List:
        """
        Find sections on the page that likely contain doctor information
        Filters out service listings and other non-doctor content
        
        Args:
            soup: BeautifulSoup object of the webpage
            
        Returns:
            List of BeautifulSoup elements containing doctor information
        """
        sections = []
        
        # Exclude navigation, menu, footer, and service-related sections
        exclude_patterns = [
            {'class': re.compile(r'nav|menu|footer|header|sidebar|service|services', re.I)},
            {'id': re.compile(r'nav|menu|footer|header|sidebar|service|services', re.I)},
        ]
        
        exclude_ids = set()
        exclude_classes = set()
        for pattern in exclude_patterns:
            if 'class' in pattern:
                found = soup.find_all(['div', 'section', 'article', 'nav'], **pattern)
                for elem in found:
                    classes = elem.get('class', [])
                    exclude_classes.update(classes)
            if 'id' in pattern:
                found = soup.find_all(['div', 'section', 'article', 'nav'], **pattern)
                for elem in found:
                    elem_id = elem.get('id', '')
                    if elem_id:
                        exclude_ids.add(elem_id)
        
        # Look for common class names and IDs related to doctors
        patterns = [
            {'class': re.compile(r'doctor|dentist|physician|team|staff|provider|member', re.I)},
            {'id': re.compile(r'doctor|dentist|physician|team|staff|provider|member', re.I)},
        ]
        
        for pattern in patterns:
            found = soup.find_all(['div', 'section', 'article'], **pattern)
            for elem in found:
                # Skip if in excluded sections
                elem_id = elem.get('id', '')
                elem_classes = elem.get('class', [])
                if elem_id in exclude_ids:
                    continue
                if any(cls in exclude_classes for cls in elem_classes):
                    continue
                
                # Check text content for doctor-related keywords
                text = elem.get_text().lower()
                doctor_keywords = ['dr.', 'doctor', 'dentist', 'dds', 'dmd', 'graduated', 
                                 'education', 'bio', 'hometown', 'residency']
                service_keywords = ['services', 'preventive dentistry', 'dental fillings', 
                                  'cosmetic dentistry', 'orthodontics']
                
                has_doctor = any(kw in text for kw in doctor_keywords)
                has_service_only = any(kw in text for kw in service_keywords) and not has_doctor
                
                if has_doctor and not has_service_only:
                    sections.append(elem)
        
        # Also look for cards or items that might contain doctor info
        card_patterns = [
            {'class': re.compile(r'card|item|member|profile|bio|person', re.I)},
        ]
        
        for pattern in card_patterns:
            cards = soup.find_all(['div', 'article', 'li'], **pattern)
            # Filter cards that might contain doctor info
            for card in cards:
                # Skip excluded sections
                card_id = card.get('id', '')
                card_classes = card.get('class', [])
                if card_id in exclude_ids:
                    continue
                if any(cls in exclude_classes for cls in card_classes):
                    continue
                
                text = card.get_text().lower()
                
                # Must have doctor keywords
                if not any(keyword in text for keyword in ['dr.', 'doctor', 'dentist', 'dds', 'dmd']):
                    continue
                
                # Exclude if it's clearly a service listing
                if any(keyword in text for keyword in ['preventive dentistry', 'dental fillings',
                                                      'cosmetic dentistry', 'services']):
                    # Only exclude if it doesn't also have doctor name
                    if not re.search(r'dr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+', text, re.I):
                        continue
                
                sections.append(card)
        
        return sections
    
    def extract_multiple_doctors_from_page(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """
        Extract multiple doctors from a page by finding all doctor name patterns
        and extracting content around each one
        
        Args:
            soup: BeautifulSoup object of the webpage
            base_url: Base URL of the website
            
        Returns:
            List of doctor dictionaries
        """
        doctors = []
        if not soup:
            return doctors
        
        # Find all doctor name patterns in the text
        text = soup.get_text()
        
        # Pattern to find doctor names: "Dr. FirstName LastName" or "Doctor FirstName LastName"
        doctor_name_patterns = [
            r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'Doctor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        
        found_names = set()
        for pattern in doctor_name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                full_match = match.group(0)  # "Dr. John Hisel"
                name_only = match.group(1).strip()  # "John Hisel"
                
                # Clean up name - remove any extra whitespace or trailing text
                name_only = re.sub(r'\s+', ' ', name_only)  # Normalize whitespace
                name_only = re.sub(r'\s+Dr\s*$', '', name_only, flags=re.I)  # Remove trailing "Dr"
                name_only = name_only.strip()
                
                name_key = name_only.lower()
                
                # Skip if we've already found this doctor
                if name_key in found_names:
                    continue
                
                # Validate the name
                if not self.is_valid_doctor_name(name_only):
                    continue
                
                found_names.add(name_key)
                
                # Create doctor entry - we'll extract bio from surrounding text
                doctor = {
                    'name': name_only,
                    'bio': '',
                    'age': None,
                    'hometown': '',
                    'education': '',
                    'photo_url': ''
                }
                
                # Try to find the HTML element containing this doctor's info
                doctor_elem = self.find_element_containing_text(soup, full_match)
                
                if doctor_elem:
                    # Extract bio from the element
                    bio_text = doctor_elem.get_text(separator=' ', strip=True)
                    bio_text = ' '.join(bio_text.split())
                    doctor['bio'] = bio_text[:2000] if len(bio_text) > 2000 else bio_text
                    
                    # Try to extract photo from element
                    doctor['photo_url'] = self.extract_photo(doctor_elem, base_url)
                else:
                    # If we can't find a specific element, extract from surrounding text
                    # Find the position of the name in the full text
                    text = soup.get_text()
                    name_pos = text.find(full_match)
                    
                    if name_pos >= 0:
                        # Get text after the name (up to next doctor name or end)
                        # Look for next "Dr." to find the end of this doctor's section
                        next_dr_pattern = r'Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
                        remaining_text = text[name_pos:]
                        next_match = re.search(next_dr_pattern, remaining_text[100:])  # Skip first 100 chars to avoid matching same name
                        
                        if next_match:
                            bio_text = remaining_text[:next_match.start() + 100]
                        else:
                            # No next doctor found, take up to 2000 chars
                            bio_text = remaining_text[:2000]
                        
                        # Clean up bio text
                        bio_text = ' '.join(bio_text.split())
                        doctor['bio'] = bio_text[:2000] if len(bio_text) > 2000 else bio_text
                
                # Extract other fields from bio
                if doctor['bio']:
                    doctor['age'] = self.extract_age_from_bio(doctor['bio'])
                    doctor['hometown'] = self.extract_hometown(doctor['bio'])
                    doctor['education'] = self.extract_education(doctor['bio'])
                
                # Only add if we have at least a name and some bio content
                if doctor['name'] and len(doctor['bio']) > 10:
                    doctors.append(doctor)
                elif doctor['name']:
                    # Even with minimal bio, add the doctor
                    doctors.append(doctor)
        
        logger.info(f"Extracted {len(doctors)} doctors using name pattern matching")
        return doctors
    
    def find_element_containing_text(self, soup: BeautifulSoup, text: str) -> Optional:
        """
        Find the HTML element that contains the specified text
        Returns the most specific parent element containing the text
        
        Args:
            soup: BeautifulSoup object
            text: Text to find
            
        Returns:
            BeautifulSoup element containing the text or None
        """
        # Find all elements containing this text
        elements = soup.find_all(string=re.compile(re.escape(text[:20]), re.I))
        
        if not elements:
            return None
        
        # Get the parent element of the first match
        # Look for a meaningful container (div, article, section, etc.)
        for text_node in elements:
            parent = text_node.parent
            max_depth = 5
            depth = 0
            
            # Traverse up to find a good container
            while parent and depth < max_depth:
                tag_name = parent.name.lower()
                
                # Look for container-like elements
                if tag_name in ['div', 'article', 'section', 'li', 'td']:
                    # Check if this element seems to contain doctor info
                    parent_text = parent.get_text()
                    if len(parent_text) > 50:  # Has substantial content
                        return parent
                
                parent = parent.parent
                depth += 1
            
            # If no good container found, return the immediate parent
            return text_node.parent
        
        return None
    
    def extract_single_doctor(self, soup_element, base_url: str) -> Optional[Dict]:
        """
        Extract information for a single doctor from a soup element
        Per requirements: Extract full name, full bio text, age, hometown, education, photo
        Includes validation to ensure it's actually a doctor profile
        
        Args:
            soup_element: BeautifulSoup element containing doctor info
            base_url: Base URL for resolving relative links
            
        Returns:
            Dictionary with doctor information or None
        """
        doctor = {
            'name': '',
            'bio': '',
            'age': None,
            'hometown': '',
            'education': '',
            'photo_url': ''
        }
        
        # Get all text from the element
        element_text = soup_element.get_text(separator=' ', strip=True)
        
        # Skip if text is too short or looks like navigation/menu
        if len(element_text.strip()) < 20:
            return None
        
        # Skip common non-doctor content patterns
        text_lower = element_text.lower()
        skip_patterns = [
            'page not found', '404', 'error', 'navigation', 'menu', 'footer',
            'contact us', 'get directions', 'phone:', 'address:', 'copyright',
            'all rights reserved', 'site by', 'privacy policy', 'terms of service'
        ]
        
        if any(pattern in text_lower for pattern in skip_patterns):
            return None
        
        # Extract name - multiple strategies
        name = self.extract_doctor_name(soup_element, element_text)
        if not name or not self.is_valid_doctor_name(name):
            return None
        
        doctor['name'] = name
        
        # Extract bio - full bio text as per requirements
        doctor['bio'] = self.extract_bio(soup_element, element_text)
        
        # Additional validation: bio should have some meaningful content
        if len(doctor['bio'].strip()) < 10:
            # If bio is too short, it might not be a real doctor profile
            return None
        
        # Extract age from graduation year (assume graduation at age 26)
        doctor['age'] = self.extract_age_from_bio(doctor['bio'])
        
        # Extract hometown (if stated)
        doctor['hometown'] = self.extract_hometown(doctor['bio'])
        
        # Extract education/dental school information
        doctor['education'] = self.extract_education(doctor['bio'])
        
        # Extract photo (optional but included)
        doctor['photo_url'] = self.extract_photo(soup_element, base_url)
        
        return doctor
    
    def is_valid_doctor_name(self, name: str) -> bool:
        """
        Validate if a name is likely a doctor name and not a service/listing
        
        Args:
            name: Name to validate
            
        Returns:
            True if likely a doctor name, False otherwise
        """
        if not name or len(name.strip()) < 3:
            return False
        
        name_lower = name.lower().strip()
        
        # Exclude service listings and common non-doctor terms
        excluded_terms = [
            'services', 'preventive', 'dentistry', 'dental', 'fillings', 'crowns',
            'cosmetic', 'restorative', 'prosthetic', 'orthodontics', 'sedation',
            'emergency', 'contact', 'lafayette', 'dentist', 'checkups', 'cleanings',
            'bruxism', 'tmj', 'snoring', 'sleep', 'apnea', 'therapy', 'canal',
            'wisdom', 'teeth', 'extraction', 'whitening', 'bonding', 'contouring',
            'gum', 'veneers', 'bridges', 'dentures', 'implants', 'invisalign',
            'aligners', 'braces', 'about us', 'page not found', 'staff', 'team',
            'our', 'welcome', 'find', 'directions', 'phone', 'address', 'rights',
            'reserved', 'site by', 'menu', 'navigation', 'home', 'services'
        ]
        
        # Check if name is in excluded terms
        if name_lower in excluded_terms:
            return False
        
        # Check if name contains excluded terms
        if any(term in name_lower for term in excluded_terms):
            return False
        
        # Exclude if it's all uppercase and looks like a service (e.g., "PREVENTIVE DENTISTRY")
        if name.isupper() and len(name.split()) > 1:
            # Allow if it looks like a doctor name pattern (DR. NAME or just NAME)
            if re.match(r'^DR\.?\s+[A-Z][A-Z\s]+$', name, re.I):
                # This is a doctor name like "DR. YOUNG"
                pass
            else:
                # Check if it contains service-like words
                service_words = ['DENTISTRY', 'SERVICES', 'PREVENTIVE', 
                               'RESTORATIVE', 'COSMETIC', 'PROSTHETIC', 'ORTHODONTICS',
                               'FILLINGS', 'CROWNS', 'IMPLANTS', 'BRACES']
                # Only exclude if it's clearly a service (has service word and is short/not a name)
                if any(word in name for word in service_words) and len(name.split()) <= 3:
                    return False
        
        # Must have at least 1 word (allow single names like "YOUNG" if uppercase with DR)
        name_parts = name.split()
        if len(name_parts) < 1:
            return False
        
        # If it's a single word and uppercase, might be a last name (allow if starts with DR)
        if len(name_parts) == 1:
            # Allow single uppercase names that might be last names
            if name.isupper() and len(name) > 3:
                return True
            return False
        
        # Should start with capital letters (proper name format) - allow all uppercase too
        if not (name_parts[0][0].isupper() or name.isupper()):
            return False
        
        # Each part should be at least 2 characters (except for single uppercase names)
        if not name.isupper() and any(len(part) < 2 for part in name_parts):
            return False
        
        return True
    
    def extract_doctor_name(self, soup_element, text: str) -> str:
        """
        Extract doctor name using multiple strategies with validation
        
        Args:
            soup_element: BeautifulSoup element containing doctor info
            text: Text content from the element
            
        Returns:
            Doctor name or empty string
        """
        name = ''
        
        # First, check if the text contains doctor-related keywords to filter out services
        text_lower = text.lower()
        doctor_keywords = ['dr.', 'doctor', 'dentist', 'dds', 'dmd', 'physician', 'dental school', 
                          'graduated', 'education', 'residency', 'bio', 'biography', 'hometown',
                          'born', 'raised', 'practice', 'team', 'staff', 'meet']
        
        # Skip if it looks like a service listing (has service keywords but no doctor keywords)
        service_keywords = ['preventive dentistry', 'restorative dentistry', 'cosmetic dentistry', 
                           'orthodontics', 'sedation dentistry', 'emergency dentistry', 
                           'dental fillings', 'dental crowns', 'dental implants']
        
        has_doctor_keyword = any(keyword in text_lower for keyword in doctor_keywords)
        has_service_phrase = any(phrase in text_lower for phrase in service_keywords)
        
        # Only skip if it's clearly a service listing (has service phrases and no doctor context)
        if has_service_phrase and not has_doctor_keyword:
            # But allow if it contains a doctor name pattern
            if not re.search(r'dr\.?\s+[A-Z]', text, re.I):
                return ''
        
        # Strategy 1: Look for name in headings or specific name classes
        name_tags = soup_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5'], 
                                         class_=re.compile(r'name|title|doctor', re.I))
        if not name_tags:
            name_tags = soup_element.find_all(['strong', 'span', 'div', 'p'], 
                                            class_=re.compile(r'name|title|doctor', re.I))
        
        if name_tags:
            for tag in name_tags:
                name = tag.get_text(strip=True)
                # Clean up common prefixes
                name = re.sub(r'^(Dr\.?|Doctor)\s+', '', name, flags=re.I)
                if self.is_valid_doctor_name(name):
                    return name
        
        # Strategy 2: Look for "Dr. FirstName LastName" pattern
        name_patterns = [
            r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'Doctor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*,\s*(?:D\.?D\.?S\.?|D\.?M\.?D\.?)',
            r'(?:D\.?D\.?S\.?|D\.?M\.?D\.?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1).strip()
                if self.is_valid_doctor_name(name):
                    return name
        
        # Strategy 3: Look for text that looks like a name (2-4 capitalized words)
        # Only if we have doctor keywords in the text
        if has_doctor_keyword:
            words = text.split()
            for i in range(len(words) - 1):
                if (words[i][0].isupper() and words[i+1][0].isupper() and 
                    len(words[i]) > 2 and len(words[i+1]) > 2):
                    potential_name = f"{words[i]} {words[i+1]}"
                    if self.is_valid_doctor_name(potential_name):
                        return potential_name
        
        return ''
    
    def extract_bio(self, soup_element, text: str) -> str:
        """
        Extract full bio text
        
        Args:
            soup_element: BeautifulSoup element containing doctor info
            text: Text content from the element
            
        Returns:
            Full bio text
        """
        # Look for bio-specific elements
        bio_elements = soup_element.find_all(['p', 'div'], class_=re.compile(r'bio|about|description|text|content', re.I))
        
        if bio_elements:
            bio_text = ' '.join([elem.get_text(strip=True) for elem in bio_elements])
        else:
            # Get all paragraph text
            paragraphs = soup_element.find_all('p')
            if paragraphs:
                bio_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                # Fallback to all text
                bio_text = text
        
        # Clean up and limit length
        bio_text = ' '.join(bio_text.split())
        # Keep full bio but limit to reasonable length (2000 chars)
        bio_text = bio_text[:2000] if len(bio_text) > 2000 else bio_text
        return bio_text
    
    def extract_age_from_bio(self, bio: str) -> Optional[int]:
        """
        Extract age from bio by finding graduation year
        Per requirements: If graduation year appears, assume dental school graduation at age 26
        Example: If graduated in 2020 → age = 26 + (current_year - 2020)
        
        Args:
            bio: Bio text containing graduation information
            
        Returns:
            Age as integer or None if not found
        """
        if not bio:
            return None
        
        # Look for graduation year patterns
        patterns = [
            r'(?:graduated|graduation|class of)\s+(?:from|in)?\s*(\d{4})',
            r'(\d{4})\s*(?:graduate|graduation)',
            r'(?:D\.?D\.?S\.?|D\.?M\.?D\.?)\s*(?:from|,)?\s*(\d{4})',
            r'(\d{4})\s*(?:D\.?D\.?S\.?|D\.?M\.?D\.?)',
            r'earned\s+(?:his|her|their)?\s*(?:D\.?D\.?S\.?|D\.?M\.?D\.?)\s+(?:in|from)?\s*(\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, bio, re.I)
            if match:
                try:
                    grad_year = int(match.group(1))
                    # Sanity check: graduation year should be reasonable
                    if 1950 <= grad_year <= self.current_year:
                        # Per requirements: assume graduation at age 26
                        age = 26 + (self.current_year - grad_year)
                        # Sanity check: age should be reasonable
                        if 25 <= age <= 80:
                            return age
                except ValueError:
                    continue
        
        return None
    
    def extract_hometown(self, bio: str) -> str:
        """
        Extract hometown from bio (if stated)
        
        Args:
            bio: Bio text containing hometown information
            
        Returns:
            Hometown string or empty string
        """
        if not bio:
            return ''
        
        patterns = [
            r'(?:from|hometown|born in|raised in|native of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:grew up in|originally from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:hails from|comes from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, bio, re.I)
            if match:
                hometown = match.group(1).strip()
                # Filter out common false positives
                if hometown.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'our', 'team']:
                    # Additional validation: should look like a place name
                    if len(hometown.split()) <= 3:  # Reasonable place name length
                        return hometown
        
        return ''
    
    def extract_education(self, bio: str) -> str:
        """
        Extract education/dental school information
        
        Args:
            bio: Bio text containing education information
            
        Returns:
            Education/dental school string or empty string
        """
        if not bio:
            return ''
        
        patterns = [
            r'(?:graduated from|attended|education|school|university|college)[:\s]+([^\.]+?)(?:\.|$)',
            r'(?:D\.?D\.?S\.?|D\.?M\.?D\.?)\s*(?:from|at)?\s*([^\.]+?)(?:\.|$)',
            r'(?:earned|received|obtained)\s+(?:his|her|their)?\s*(?:D\.?D\.?S\.?|D\.?M\.?D\.?)\s+(?:from|at)?\s*([^\.]+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, bio, re.I)
            if match:
                education = match.group(1).strip()
                # Clean up
                education = re.sub(r'^(from|at|in)\s+', '', education, flags=re.I)
                # Limit length to 200 characters
                return education[:200]
        
        return ''
    
    def extract_photo(self, soup_element, base_url: str) -> str:
        """
        Extract doctor photo URL (optional per requirements)
        
        Args:
            soup_element: BeautifulSoup element containing doctor info
            base_url: Base URL for resolving relative links
            
        Returns:
            Photo URL string or empty string
        """
        # Look for images
        img_tags = soup_element.find_all('img', src=True)
        
        for img in img_tags:
            img_src = img.get('src', '')
            img_alt = img.get('alt', '').lower()
            img_class = img.get('class', [])
            img_class_str = ' '.join(img_class).lower() if img_class else ''
            
            # Check if image might be a doctor photo
            if any(keyword in img_src.lower() or keyword in img_alt or keyword in img_class_str 
                   for keyword in ['doctor', 'dentist', 'physician', 'team', 'staff', 'profile', 'headshot', 'photo']):
                # Convert relative URL to absolute
                if img_src.startswith('/'):
                    return urljoin(base_url, img_src)
                elif img_src.startswith('http'):
                    return img_src
                else:
                    return urljoin(base_url, img_src)
        
        # If no specific doctor image found, return first image as fallback
        if img_tags:
            img_src = img_tags[0].get('src', '')
            if img_src:
                if img_src.startswith('/'):
                    return urljoin(base_url, img_src)
                elif img_src.startswith('http'):
                    return img_src
                else:
                    return urljoin(base_url, img_src)
        
        return ''
    
    def identify_owner(self, doctors: List[Dict], practice_name: str, soup: BeautifulSoup) -> List[Dict]:
        """
        Identify owner using tiered logic system per requirements:
        1. Single-doctor rule
        2. Name-match rule
        3. Weighted scoring model
        
        Args:
            doctors: List of doctor dictionaries
            practice_name: Name of the practice
            soup: BeautifulSoup object of the website
            
        Returns:
            List of doctors with 'role' field added ('Owner' or 'Associate')
        """
        if not doctors:
            return []
        
        # Step 1: Single-doctor rule
        # If only one doctor is listed → that doctor is the owner
        if len(doctors) == 1:
            doctors[0]['role'] = 'Owner'
            logger.info(f"Single doctor found - assigned as Owner: {doctors[0].get('name')}")
            return doctors
        
        # Step 2: Name-match rule
        # If the practice name contains a doctor's last name → that doctor is the owner
        practice_name_lower = practice_name.lower() if practice_name else ''
        
        for doctor in doctors:
            doctor_name_parts = doctor.get('name', '').lower().split()
            if doctor_name_parts:
                last_name = doctor_name_parts[-1]
                # Check if last name appears in practice name
                if last_name in practice_name_lower:
                    doctor['role'] = 'Owner'
                    for other_doctor in doctors:
                        if other_doctor != doctor:
                            other_doctor['role'] = 'Associate'
                    logger.info(f"Name match found - {doctor.get('name')} assigned as Owner")
                    return doctors
        
        # Step 3: Weighted scoring model
        # If multiple doctors exist and no direct match is found, compute weighted score
        logger.info("Using weighted scoring model to identify owner")
        
        # Get full website text for name prominence calculation
        if soup:
            full_text = soup.get_text().lower()
        else:
            full_text = ''
        
        scores = []
        for i, doctor in enumerate(doctors):
            score = 0
            doctor_name = doctor.get('name', '')
            doctor_name_lower = doctor_name.lower()
            
            # Factor 1: Name prominence
            # How frequently the doctor's name appears across the website
            name_count = full_text.count(doctor_name_lower)
            score += name_count * 2  # 2x weight
            
            # Factor 2: Listing order
            # Doctors listed earlier get priority weighting
            order_weight = (len(doctors) - i) * 3
            score += order_weight
            
            # Factor 3: Likely owner age range
            # Doctors aged 35–55 receive additional weight
            age = doctor.get('age')
            if age:
                if 35 <= age <= 55:
                    score += 10
            
            scores.append((score, i, doctor_name))
            logger.debug(f"  {doctor_name}: Total score = {score}")
        
        # Sort by score (highest first)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Assign roles
        # The doctor with the highest composite score is assigned as Owner
        # All others are marked as Associate
        for idx, (score, doctor_idx, doctor_name) in enumerate(scores):
            if idx == 0:
                doctors[doctor_idx]['role'] = 'Owner'
                logger.info(f"Owner identified: {doctor_name} (score: {score})")
            else:
                doctors[doctor_idx]['role'] = 'Associate'
        
        return doctors
    
    def process_website(self, website_url: str, practice_name: str = '') -> List[Dict]:
        """
        Process a single website and extract doctor information
        
        Args:
            website_url: URL of the practice website
            practice_name: Optional practice name (for owner identification)
            
        Returns:
            List of doctor records with all required fields
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing website: {website_url}")
        
        base_result = {
            'website': website_url,
            'practice_name': practice_name or '',
            'doctor_name': '',
            'bio': '',
            'age': '',
            'hometown': '',
            'education': '',
            'photo_url': '',
            'role': ''
        }
        
        # Step 1: Scrape website
        soup = self.scrape_website(website_url)
        if not soup:
            logger.warning(f"Could not scrape website: {website_url}")
            return [base_result]
        
        # Step 2: Extract doctors from website
        doctors = self.extract_doctors_from_website(soup, website_url)
        
        if not doctors:
            logger.warning(f"No doctors found on website: {website_url}")
            return [base_result]
        
        # Step 3: Identify owner using tiered logic
        doctors = self.identify_owner(doctors, practice_name, soup)
        
        # Step 4: Format output (one row per doctor)
        results = []
        for doctor in doctors:
            results.append({
                'website': website_url,
                'practice_name': practice_name or '',
                'doctor_name': doctor.get('name', ''),
                'bio': doctor.get('bio', ''),
                'age': doctor.get('age', '') if doctor.get('age') else '',
                'hometown': doctor.get('hometown', ''),
                'education': doctor.get('education', ''),
                'photo_url': doctor.get('photo_url', ''),
                'role': doctor.get('role', '')  # "Owner" or "Associate"
            })
        
        logger.info(f"Successfully processed {website_url}: {len(results)} doctor(s) found")
        return results
    
    def run(self, input_csv: str = 'req_input.csv', output_file: str = 'output.csv'):
        """
        Main method to read URLs from CSV and scrape doctor data
        
        Args:
            input_csv: Path to input CSV file containing URLs
            output_file: Path to output CSV file
        """
        logger.info("Starting Doctor Data Scraper")
        logger.info(f"Input file: {input_csv}")
        logger.info(f"Output file: {output_file}")
        
        # Read URLs from CSV
        urls = self.read_input_csv(input_csv)
        
        if not urls:
            logger.error("No URLs found in input file. Exiting.")
            return
        
        # Process each URL
        all_results = []
        total_urls = len(urls)
        
        for idx, url in enumerate(urls, 1):
            logger.info(f"\nProcessing URL {idx}/{total_urls}")
            try:
                results = self.process_website(url)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error processing website {url}: {e}")
                # Add error row
                all_results.append({
                    'website': url,
                    'practice_name': '',
                    'doctor_name': '',
                    'bio': '',
                    'age': '',
                    'hometown': '',
                    'education': '',
                    'photo_url': '',
                    'role': f'ERROR: {str(e)[:50]}'
                })
            
            # Minimal delay between websites
            time.sleep(0.3)
        
        # Create output DataFrame
        output_df = pd.DataFrame(all_results)
        
        # Write to output file
        output_ext = os.path.splitext(output_file)[1].lower()
        
        if output_ext == '.xlsx':
            # Write to Excel file
            try:
                output_df.to_excel(output_file, index=False, engine='openpyxl')
                logger.info(f"Results written to Excel file: {output_file}")
            except Exception as e:
                logger.error(f"Error writing to Excel file: {e}")
                # Fallback to CSV
                csv_file = output_file.replace('.xlsx', '.csv')
                output_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                logger.info(f"Results written to CSV file: {csv_file}")
        else:
            # Write to CSV file
            try:
                output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"Results written to CSV file: {output_file}")
            except Exception as e:
                logger.error(f"Error writing to CSV file: {e}")
                raise
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping complete!")
        logger.info(f"Total URLs processed: {total_urls}")
        logger.info(f"Total doctor records created: {len(all_results)}")
        logger.info(f"Results written to: {output_file}")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution function"""
    # Ollama Configuration
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')  # Default to qwen2.5:7b-instruct, can override with env var
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    # Initialize scraper with Ollama
    scraper = DoctorScraper(ollama_model=OLLAMA_MODEL, ollama_base_url=OLLAMA_BASE_URL)
    
    # Configuration
    INPUT_CSV = 'req_input.csv'  # Will also try Req_Inpu.csv
    OUTPUT_FILE = 'doctor_data_output_llm.csv'  # Different filename to distinguish from rule-based output
    
    # If output file is locked, try alternative name
    if os.path.exists(OUTPUT_FILE):
        try:
            # Try to open it to check if it's locked
            with open(OUTPUT_FILE, 'a'):
                pass
        except (PermissionError, IOError):
            # File is locked, use alternative name
            OUTPUT_FILE = 'doctor_data_output_llm_new.csv'
            logger.warning(f"Original output file is locked, using: {OUTPUT_FILE}")
    
    # Run the scraper
    scraper.run(input_csv=INPUT_CSV, output_file=OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("Doctor Data Scraping with LLM completed successfully!")
    print("="*60)
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print(f"Check '{OUTPUT_FILE}' for results")
    print(f"Check 'scraper.log' for detailed logs")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

