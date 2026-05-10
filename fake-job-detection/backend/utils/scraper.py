"""
Web Scraper Module - Extract job data from URLs
Handles multiple job board formats: LinkedIn, Indeed, Generic HTML
"""

import re
import logging
from typing import Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def scrape_job_url(url: str) -> Optional[Dict]:
    """
    Scrape job posting data from a URL.
    Tries requests+BeautifulSoup, falls back to basic URL analysis.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        domain = urlparse(url).netloc.lower()

        # Try domain-specific scrapers first
        if "linkedin.com" in domain:
            return scrape_linkedin(soup, url)
        elif "indeed.com" in domain:
            return scrape_indeed(soup, url)
        elif "glassdoor.com" in domain:
            return scrape_glassdoor(soup, url)
        else:
            return scrape_generic(soup, url, response.text)

    except ImportError:
        logger.warning("requests/beautifulsoup4 not installed, using URL-based analysis")
        return analyze_url_only(url)
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return analyze_url_only(url)


def scrape_linkedin(soup, url: str) -> Dict:
    """Scrape LinkedIn job postings"""
    job = {"source_url": url, "platform": "LinkedIn"}

    job["title"] = _get_text(soup, [
        ".job-details-jobs-unified-top-card__job-title",
        ".jobs-unified-top-card__job-title",
        "h1.t-24"
    ])

    job["company"] = _get_text(soup, [
        ".job-details-jobs-unified-top-card__company-name",
        ".jobs-unified-top-card__company-name",
        ".topcard__org-name-link"
    ])

    job["location"] = _get_text(soup, [
        ".job-details-jobs-unified-top-card__bullet",
        ".jobs-unified-top-card__bullet",
        ".topcard__flavor--bullet"
    ])

    job["description"] = _get_text(soup, [
        ".jobs-description__content",
        ".show-more-less-html__markup",
        "[class*='description']"
    ])

    return _fill_from_meta(job, soup)


def scrape_indeed(soup, url: str) -> Dict:
    """Scrape Indeed job postings"""
    job = {"source_url": url, "platform": "Indeed"}

    job["title"] = _get_text(soup, [
        ".jobsearch-JobInfoHeader-title",
        "[class*='JobTitle']",
        "h1.icl-u-xs-mb--xs"
    ])

    job["company"] = _get_text(soup, [
        ".jobsearch-CompanyInfoWithoutHeaderImage",
        "[data-testid='inlineHeader-companyName']",
        ".icl-u-lg-mr--sm"
    ])

    job["location"] = _get_text(soup, [
        ".jobsearch-JobInfoHeader-subtitle > div:last-child",
        "[data-testid='job-location']"
    ])

    job["salary"] = _get_text(soup, [
        ".jobsearch-JobMetadataHeader-item",
        "[class*='salary']",
        "#salaryInfoAndJobType"
    ])

    job["description"] = _get_text(soup, [
        "#jobDescriptionText",
        ".jobsearch-jobDescriptionText"
    ])

    return _fill_from_meta(job, soup)


def scrape_glassdoor(soup, url: str) -> Dict:
    """Scrape Glassdoor job postings"""
    job = {"source_url": url, "platform": "Glassdoor"}

    job["title"] = _get_text(soup, ["[data-test='job-title']", ".jobTitle", "h1"])
    job["company"] = _get_text(soup, ["[data-test='employer-name']", ".employerName"])
    job["location"] = _get_text(soup, ["[data-test='location']", ".location"])
    job["salary"] = _get_text(soup, ["[data-test='detailSalary']", ".salary"])
    job["description"] = _get_text(soup, ["[class*='jobDescription']", ".desc"])

    return _fill_from_meta(job, soup)


def scrape_generic(soup, url: str, raw_html: str) -> Dict:
    """Generic scraper for unknown job boards"""
    job = {"source_url": url, "platform": "Unknown"}

    # Try schema.org structured data first
    import json as json_lib
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        try:
            data = json_lib.loads(script.string)
            if isinstance(data, list):
                data = data[0]
            if data.get("@type") in ["JobPosting", "jobPosting"]:
                job["title"] = data.get("title", "")
                job["company"] = data.get("hiringOrganization", {}).get("name", "")
                job["location"] = data.get("jobLocation", {}).get("address", {}).get("addressLocality", "")
                job["salary"] = str(data.get("baseSalary", ""))
                job["description"] = _clean_html(str(data.get("description", "")))
                job["requirements"] = data.get("qualifications", "")
                return job
        except Exception:
            continue

    # Heuristic extraction
    job["title"] = _get_text(soup, [
        "h1", ".job-title", ".position-title", "[class*='title']", "[class*='job']"
    ])

    job["company"] = _get_text(soup, [
        ".company-name", ".employer", "[class*='company']", "[class*='employer']"
    ])

    job["salary"] = _extract_salary_from_text(soup.get_text())

    # Get main content area
    main_content = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"job|description|content", re.I)) or
        soup.find(class_=re.compile(r"job|description|content", re.I))
    )

    if main_content:
        job["description"] = main_content.get_text(separator=" ", strip=True)[:3000]
    else:
        # Extract all visible text
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        job["description"] = soup.get_text(separator=" ", strip=True)[:3000]

    return _fill_from_meta(job, soup)


def analyze_url_only(url: str) -> Dict:
    """Minimal analysis when scraping fails — analyze URL structure itself"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    from models.detector import TRUSTED_DOMAINS
    is_trusted = any(td in domain for td in TRUSTED_DOMAINS)

    return {
        "source_url": url,
        "platform": domain,
        "title": "",
        "company": "",
        "description": f"Job posting from {domain}. Web scraping unavailable.",
        "requirements": "",
        "salary": "",
        "location": "",
        "full_text": f"Job posting URL: {url}. Domain: {domain}. Trusted platform: {is_trusted}",
        "scraping_failed": True
    }


def _get_text(soup, selectors: list) -> str:
    """Try multiple CSS selectors, return first match"""
    for selector in selectors:
        try:
            el = soup.select_one(selector)
            if el:
                return el.get_text(strip=True)
        except Exception:
            continue
    return ""


def _fill_from_meta(job: Dict, soup) -> Dict:
    """Fill missing fields from meta tags"""
    if not job.get("title"):
        meta = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
        if meta:
            job["title"] = meta.get("content", "")

    if not job.get("description"):
        meta = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if meta:
            job["description"] = meta.get("content", "")

    # Build full_text
    parts = [job.get(k, "") for k in ["title", "company", "location", "salary", "description", "requirements"]]
    job["full_text"] = " ".join(p for p in parts if p)

    return job


def _clean_html(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text).strip()


def _extract_salary_from_text(text: str) -> str:
    patterns = [
        r'\$[\d,]+(?:\s*-\s*\$[\d,]+)?(?:\s*(?:per|/)\s*(?:year|yr|hour|hr|month))?',
        r'[\d,]+(?:\s*-\s*[\d,]+)?\s*(?:USD|EUR|GBP)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return ""
