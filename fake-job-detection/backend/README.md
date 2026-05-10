# Fake Job Detection System - Backend

## Setup

### Prerequisites
- Python 3.9+
- Tesseract OCR (for image analysis)

### Install Tesseract
**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Run the API Server
```bash
python main.py
# OR
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/analyze/image | Analyze job posting image |
| POST | /api/v1/analyze/url | Analyze job posting URL |
| POST | /api/v1/analyze/text | Analyze raw job text |
| GET | /api/v1/model/stats | Get model statistics |
| GET | /health | Health check |

## Model Architecture

The hybrid AI model combines:
1. **Random Forest** - Structural feature analysis (missing fields, text length, formatting)
2. **Logistic Regression** - Linear text feature scoring (linguistic patterns)
3. **NLP Analysis** - Deep pattern matching (BERT-like keyword understanding)

Ensemble weighting: RF(40%) + LR(30%) + NLP(30%)
