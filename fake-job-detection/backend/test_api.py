"""
Test script for Fake Job Detection API
Run after starting the backend: python main.py
"""

import requests
import json

BASE = "http://localhost:8000/api/v1"

def test_fake_job():
    """Test with a clearly fake job"""
    print("\n=== TEST 1: Fake Job (Text) ===")
    response = requests.post(f"{BASE}/analyze/text", json={
        "title": "WORK FROM HOME EARN $5000 WEEKLY!!!",
        "company": "",  # No company
        "description": "URGENT HIRING! No experience needed! Earn $5000 per week working from home! Just pay registration fee of $99 to start. Apply immediately! Limited slots! 100% guaranteed income! Send your bank details to start!",
        "requirements": "Anyone can do it! No experience required!",
        "salary": "Unlimited income potential",
        "location": ""
    })
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_score']}%")
    print(f"Risk Level: {result['risk_level']['label']}")
    print(f"Top Reason: {result['explanation']['reasons'][0] if result['explanation']['reasons'] else 'N/A'}")
    print(f"Highlighted Keywords: {result['explanation']['highlighted_keywords'][:5]}")


def test_real_job():
    """Test with a legitimate job"""
    print("\n=== TEST 2: Real Job (Text) ===")
    response = requests.post(f"{BASE}/analyze/text", json={
        "title": "Senior Software Engineer",
        "company": "TechCorp Inc.",
        "description": "We are looking for an experienced Software Engineer to join our engineering team. You will design and build scalable web applications. We offer competitive salary, health insurance, 401k matching, and generous PTO. Equal opportunity employer.",
        "requirements": "5+ years of software engineering experience. Bachelor's degree in Computer Science or related field. Experience with Python, JavaScript, and cloud platforms. Strong problem-solving skills required.",
        "salary": "$120,000 - $150,000 per year",
        "location": "San Francisco, CA (Hybrid)"
    })
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_score']}%")
    print(f"Risk Level: {result['risk_level']['label']}")
    print(f"Legitimate Signals: {result['explanation']['legitimate_signals'][:5]}")


def test_url():
    """Test URL analysis"""
    print("\n=== TEST 3: URL Analysis ===")
    response = requests.post(f"{BASE}/analyze/url", json={
        "url": "https://www.linkedin.com/jobs/view/software-engineer"
    })
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_score']}%")
    print(f"Input Type: {result.get('input_type')}")


def test_model_stats():
    """Get model stats"""
    print("\n=== Model Statistics ===")
    response = requests.get(f"{BASE}/model/stats")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("Fake Job Detection API - Test Suite")
    print("====================================")
    
    # Check health first
    try:
        r = requests.get("http://localhost:8000/health")
        if r.status_code == 200:
            print("✅ API is running!")
        test_fake_job()
        test_real_job()
        test_model_stats()
    except requests.ConnectionError:
        print("❌ API not running. Start with: cd backend && python main.py")
