"""
Preprocessor - Convert raw text into structured job data format
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def preprocess_job_data(text: str = "", job_dict: Dict = None) -> Dict:
    """Convert raw OCR text or dict into unified structured format"""
    if job_dict:
        return _normalize_dict(job_dict)

    if text:
        return _parse_from_text(text)

    return {}


def _parse_from_text(text: str) -> Dict:
    """Parse unstructured text into job fields using heuristics"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    job = {
        "title": "",
        "company": "",
        "location": "",
        "salary": "",
        "description": text,
        "requirements": "",
        "full_text": text
    }

    # Extract salary
    salary_pattern = r'\$[\d,]+(?:\s*[-–]\s*\$?[\d,]+)?(?:\s*(?:per|/)\s*(?:year|yr|hour|hr|month|mo|week|wk))?'
    salary_match = re.search(salary_pattern, text, re.IGNORECASE)
    if salary_match:
        job["salary"] = salary_match.group(0)

    # Extract email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        job["contact_email"] = email_match.group(0)

    # Extract phone
    phone_match = re.search(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    if phone_match:
        job["contact_phone"] = phone_match.group(0)

    # Extract location (cities, states)
    location_patterns = [
        r'\b(?:Remote|Work from Home|WFH)\b',
        r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, ST
        r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # City, Country
    ]
    for pattern in location_patterns:
        loc_match = re.search(pattern, text)
        if loc_match:
            job["location"] = loc_match.group(0)
            break

    # Try to identify job title (first prominent line)
    if lines:
        # Title is usually the first non-trivial line
        for line in lines[:5]:
            if 5 < len(line) < 80 and not re.match(r'^(http|www|\d)', line):
                job["title"] = line
                break

    # Find company name patterns
    company_patterns = [
        r'(?:Company|Employer|Organization):\s*(.+)',
        r'(?:at|@)\s+([A-Z][A-Za-z\s&,\.]+(?:Inc|LLC|Ltd|Corp|Co\.?)?)',
        r'([A-Z][A-Za-z\s&]+(?:Inc|LLC|Ltd|Corp|Co)\.?)\s+is\s+(?:hiring|looking)',
    ]
    for pattern in company_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            job["company"] = match.group(1).strip()
            break

    # Extract requirements section
    req_section = re.search(
        r'(?:Requirements?|Qualifications?|What we(?:\'re)? looking for)[:\s]*\n((?:.+\n?){1,10})',
        text, re.IGNORECASE
    )
    if req_section:
        job["requirements"] = req_section.group(1).strip()

    return job


def _normalize_dict(job_dict: Dict) -> Dict:
    """Normalize a job dict to unified format"""
    normalized = {
        "title": str(job_dict.get("title", "")),
        "company": str(job_dict.get("company", "")),
        "location": str(job_dict.get("location", "")),
        "salary": str(job_dict.get("salary", "")),
        "description": str(job_dict.get("description", "")),
        "requirements": str(job_dict.get("requirements", "")),
        "source_url": str(job_dict.get("source_url", "")),
        "platform": str(job_dict.get("platform", "")),
    }

    # Build full_text
    parts = [normalized[k] for k in ["title", "company", "location", "salary", "description", "requirements"]]
    normalized["full_text"] = " ".join(p for p in parts if p)

    return normalized
