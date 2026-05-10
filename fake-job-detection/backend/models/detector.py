"""
Hybrid AI Fake Job Detector — v2 (Trained ML Edition)
=====================================================
Fixes the "same output for all inputs" bug by replacing hardcoded
rule-based scoring with a TRAINED scikit-learn ensemble pipeline.

Changes from v1:
  1. RandomForest + LogisticRegression trained on EMSCAD-style samples
  2. TF-IDF on full job text (bigrams, 500 features) combined with
     17 structured features
  3. Model auto-trains and saves to trained_model.pkl on first run
  4. Rule-based fallback retained if sklearn unavailable
  5. Intercept bias removed — predictions now depend on actual content
"""

import re
import os
import math
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).parent / "trained_model.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA  (EMSCAD-style synthetic dataset, 35 labelled samples)
# label 1 = FAKE,  label 0 = REAL
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_SAMPLES = [
    ({"title":"Data Entry Specialist","company":"","description":"Earn $500 per day working from home! No experience required. Urgent openings available. Apply immediately! Registration fee of $50 required.","requirements":"","salary":"$500/day","location":"Remote"}, 1),
    ({"title":"Work From Home Opportunity","company":"","description":"Make $10000 a week with our proven system! Easy money, simple tasks. Anyone can do it! Send bank details to get started. Limited time offer!","requirements":"","salary":"Unlimited income","location":""}, 1),
    ({"title":"Online Marketing Manager","company":"Confidential Company","description":"Be your own boss! Earn 50000 monthly. No experience needed high salary. Pay a startup fee to unlock training.","requirements":"None","salary":"","location":""}, 1),
    ({"title":"Customer Service Rep","company":"","description":"WORK FROM HOME!! EARN BIG $$$$ DAILY!!! Click here now. Risk free opportunity. 100% guaranteed income. Secret method revealed!","requirements":"","salary":"$$$$$","location":"Anywhere"}, 1),
    ({"title":"Brand Ambassador","company":"Name withheld","description":"Flexible hours, easy money. We need people urgently. Act fast today only. Provide SSN and bank account to verify identity.","requirements":"None required","salary":"Depends","location":""}, 1),
    ({"title":"Virtual Assistant","company":"","description":"Simple tasks from home. Earn $300 daily. No experience required. Pay equipment fee of $99 to start. Don't wait apply immediately!","requirements":"Anyone can do it","salary":"$300 daily","location":"Remote"}, 1),
    ({"title":"Recruiter","company":"Anonymous employer","description":"Western Union payment method available. Data entry from home. Envelope stuffing pays $400/day. Startup opportunity unlimited income guaranteed.","requirements":"","salary":"$400/day","location":""}, 1),
    ({"title":"Sales Executive","company":"","description":"Urgent! Make money fast. Send passport copy to verify. Investment required upfront. This is a risk free work-from-home role.","requirements":"Click ads daily","salary":"Unlimited","location":"Work from home"}, 1),
    ({"title":"Administrative Assistant","company":"Newly established firm","description":"Flexible part-time work. Processing fee of $25. No experience needed. Earn thousands weekly. Hurry limited spots!","requirements":"","salary":"Thousands weekly","location":"Remote"}, 1),
    ({"title":"Content Writer","company":"","description":"MAKE MONEY NOW!!!! Easy online work. Pay application fee. $500 per day guaranteed. No skills needed. Secret method proven system.","requirements":"NONE","salary":"$500/day","location":""}, 1),
    ({"title":"Account Manager","company":"Undisclosed company","description":"100% guaranteed income. Work from home earn thousands. Registration fee required. Act fast right now. Limited time opening.","requirements":"","salary":"","location":"Online"}, 1),
    ({"title":"HR Recruiter","company":"","description":"Earn $5000 weekly from home. Provide credit card number for training materials. Background check fee and kit fee required. No experience needed high salary.","requirements":"Anyone","salary":"$5000/week","location":""}, 1),
    ({"title":"Operations Coordinator","company":"Startup opportunity","description":"Money gram payment. Wire transfer fee. Be your own boss. Unlimited income. Don't wait, apply immediately. Confidential company hiring now.","requirements":"No degree required","salary":"High pay","location":"Remote"}, 1),
    ({"title":"Part Time Job","company":"","description":"Click ads from home and earn $200/hour. No experience required. Guaranteed income risk free. $$$ daily. Hurry today only.","requirements":"","salary":"$200/hour","location":""}, 1),
    ({"title":"Social Media Specialist","company":"","description":"Work from home earn thousands daily. Easy money, flexible hours, be your own boss. Send routing number to get started. URGENT positions open!!!","requirements":"None","salary":"Earn thousands","location":"Anywhere"}, 1),
    ({"title":"Software Developer","company":"TechCorp Innovations","description":"We are urgently hiring developers. No experience required. Earn $10000 a week. Pay a small training fee to start. Flexible work from home.","requirements":"None anyone can do it","salary":"$10000/week","location":"Remote"}, 1),
    ({"title":"Finance Analyst","company":"","description":"Competitive salary offered. Background check required. However, applicants must provide bank account and social security upfront. Western union payment available. Earn $5000 weekly guaranteed.","requirements":"University degree preferred","salary":"Unlimited income","location":"New York"}, 1),
    ({"title":"Project Manager","company":"undisclosed","description":"We offer 401k and health insurance. But first, send passport and pay startup fee of $150. Urgent positions available. Today only. Earn $3000 per day.","requirements":"PTO available","salary":"Guaranteed $3000/day","location":""}, 1),
    ({"title":"Software Engineer","company":"Google LLC","description":"We are seeking an experienced software engineer to join our infrastructure team. You will design scalable backend systems, participate in code reviews, and mentor junior engineers. We offer competitive benefits, health insurance, and 401k.","requirements":"Bachelor's in Computer Science, 3+ years of experience with Python or Go, experience with distributed systems","salary":"$120,000 - $180,000/year","location":"Mountain View, CA"}, 0),
    ({"title":"Marketing Manager","company":"Procter & Gamble","description":"Join our brand team to lead digital marketing campaigns across multiple product lines. You will work cross-functionally with product and sales teams. Career growth opportunities and professional development programs available.","requirements":"Master's in Marketing or MBA, 5 years of experience, strong analytical skills","salary":"$85,000 - $105,000/year","location":"Cincinnati, OH"}, 0),
    ({"title":"Data Scientist","company":"Amazon","description":"We are hiring a data scientist to build machine learning models for supply chain optimization. You will work in a team environment with access to massive datasets. Performance review cycle twice a year.","requirements":"PhD or Master's in Statistics, experience with Python and SQL, knowledge of ML frameworks","salary":"$130,000 - $160,000/year","location":"Seattle, WA"}, 0),
    ({"title":"Product Designer","company":"Figma Inc","description":"Design intuitive user experiences for our collaborative platform. You will lead design sprints, conduct user research, and work closely with engineering. We are an equal opportunity employer with paid time off and full benefits.","requirements":"Portfolio required, 4+ years of UX design experience, proficiency in Figma","salary":"$110,000 - $145,000/year","location":"San Francisco, CA"}, 0),
    ({"title":"Financial Analyst","company":"JPMorgan Chase","description":"Analyze market data, build financial models, and support investment decisions. Competitive benefits package including health insurance, 401k, and PTO. Formal onboarding program and HR department support.","requirements":"Bachelor's in Finance or Economics, CFA preferred, years of experience in banking","salary":"$75,000 - $95,000/year","location":"New York, NY"}, 0),
    ({"title":"DevOps Engineer","company":"Microsoft","description":"Build and maintain CI/CD pipelines for Azure cloud services. Participate in on-call rotations. We provide drug test, background check required, and a full onboarding experience.","requirements":"Bachelor's degree, 3 years of DevOps experience, Kubernetes and Terraform skills","salary":"$115,000 - $145,000/year","location":"Redmond, WA"}, 0),
    ({"title":"Account Executive","company":"Salesforce","description":"Drive new business by selling CRM solutions to enterprise clients. Quota-carrying role with strong base salary plus commission. Career growth into senior AE or management track. Equal opportunity employer.","requirements":"3 years of B2B sales experience, Salesforce knowledge preferred, bachelor's degree","salary":"$80,000 base + commission","location":"Chicago, IL"}, 0),
    ({"title":"HR Business Partner","company":"Johnson & Johnson","description":"Partner with business leaders to drive talent strategy, performance management, and employee engagement. Professional development budget provided. HR department of 200+ professionals.","requirements":"Bachelor's in HR or Business, 5 years of HR experience, SHRM certification preferred","salary":"$90,000 - $110,000/year","location":"New Brunswick, NJ"}, 0),
    ({"title":"Mechanical Engineer","company":"Tesla Inc","description":"Design and validate battery enclosure systems. Work in a fast-paced team environment with cutting-edge technology. Interview process includes technical screens and on-site design review.","requirements":"BS in Mechanical Engineering, experience with SolidWorks, 2+ years of experience","salary":"$95,000 - $130,000/year","location":"Fremont, CA"}, 0),
    ({"title":"Registered Nurse","company":"Mayo Clinic","description":"Provide compassionate patient care in our cardiology unit. Full-time position with health insurance, paid time off, and tuition reimbursement. Comprehensive onboarding and mentorship program.","requirements":"BSN required, active RN license, 2 years of clinical experience, BLS certification","salary":"$65,000 - $85,000/year","location":"Rochester, MN"}, 0),
    ({"title":"Content Strategist","company":"HubSpot","description":"Develop and execute content marketing strategy across blog, email, and social channels. Collaborative team environment, flexible working, and performance review bi-annually.","requirements":"Bachelor's in Journalism or Communications, 3 years of content experience, SEO knowledge","salary":"$70,000 - $90,000/year","location":"Boston, MA"}, 0),
    ({"title":"Security Analyst","company":"Palo Alto Networks","description":"Monitor SOC alerts, investigate incidents, and improve threat detection capabilities. Background check required. We are an equal opportunity employer with full benefits.","requirements":"CompTIA Security+ or CISSP, 3 years of security operations experience, SIEM expertise","salary":"$100,000 - $130,000/year","location":"Santa Clara, CA"}, 0),
    ({"title":"Supply Chain Coordinator","company":"Unilever","description":"Coordinate inbound and outbound logistics across multiple distribution centers. Cross-functional collaboration with procurement and manufacturing teams. PTO and 401k benefits.","requirements":"Bachelor's in Supply Chain or Business, 2 years of logistics experience, SAP knowledge","salary":"$55,000 - $70,000/year","location":"Englewood Cliffs, NJ"}, 0),
    ({"title":"Graphic Designer","company":"Adobe Systems","description":"Create visual assets for marketing campaigns and product interfaces. Portfolio-based interview process. Equal opportunity employer offering full health insurance and 401k.","requirements":"BFA or equivalent, 3 years of Adobe Creative Suite experience, brand identity portfolio","salary":"$75,000 - $95,000/year","location":"San Jose, CA"}, 0),
    ({"title":"Python Developer","company":"Stripe","description":"Build internal tools and APIs for our payments infrastructure team. Code reviews, pair programming, and a strong engineering culture. Competitive salary with equity.","requirements":"Strong Python skills, REST API design experience, Bachelor's degree preferred, 2+ years","salary":"$125,000 - $155,000/year","location":"South San Francisco, CA"}, 0),
    ({"title":"Remote Customer Support","company":"Shopify","description":"Help merchants resolve account and billing issues via chat. Work from home permanently. Flexible hours within shift windows. Health insurance and paid time off included. Formal onboarding week.","requirements":"Excellent communication skills, 1 year of support experience, reliable internet","salary":"$45,000 - $55,000/year","location":"Remote (USA)"}, 0),
    ({"title":"Social Media Manager","company":"Glossier","description":"Manage Instagram and TikTok presence for our beauty brand. Creative role with room for growth. Team environment and career growth path to Senior Manager. Annual performance review.","requirements":"2 years of social media management, strong writing skills, content creation portfolio","salary":"$60,000 - $75,000/year","location":"New York, NY"}, 0),
]

# ─────────────────────────────────────────────────────────────────────────────
# FRAUD PATTERN KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
SUSPICIOUS_KEYWORDS = {
    "urgency": {"keywords": ["urgent","immediate","asap","right now","today only","limited time","act fast","don't wait","hurry","apply immediately"], "weight": 0.8, "label": "Urgency Pressure"},
    "money_upfront": {"keywords": ["registration fee","training fee","deposit required","pay to work","investment required","startup fee","application fee","processing fee","background check fee","kit fee","equipment fee"], "weight": 1.0, "label": "Payment Required (Red Flag)"},
    "unrealistic_salary": {"keywords": ["earn $5000 weekly","make $10000 a week","unlimited income","no experience needed high salary","work from home earn thousands","$500 per day","earn 50000 monthly","guaranteed income","$10000 a week","$3000 per day","earn thousands"], "weight": 0.9, "label": "Unrealistic Salary Claims"},
    "vague_description": {"keywords": ["easy money","work from home","be your own boss","flexible hours","no experience required","anyone can do it","simple tasks","data entry from home","envelope stuffing","click ads"], "weight": 0.6, "label": "Vague/Generic Description"},
    "personal_info": {"keywords": ["send bank details","provide ssn","social security","wire transfer","western union","money gram","send passport","credit card number","bank account","routing number"], "weight": 1.0, "label": "Suspicious Information Requests"},
    "poor_quality": {"keywords": ["!!!!","$$$$","click here now","100% guaranteed","risk free","no risk","secret method","proven system"], "weight": 0.7, "label": "Poor Quality Indicators"},
    "fake_company": {"keywords": ["newly established","startup opportunity","anonymous employer","confidential company","name withheld","undisclosed company"], "weight": 0.75, "label": "Suspicious Company Info"},
}

LEGITIMATE_INDICATORS = [
    "equal opportunity employer","background check required","drug test","competitive benefits",
    "health insurance","401k","pto","paid time off","university degree","bachelor's","master's",
    "years of experience","interview process","hr department","onboarding","performance review",
    "team environment","career growth","professional development",
]

TRUSTED_DOMAINS = ["linkedin.com","indeed.com","glassdoor.com","monster.com","careerbuilder.com","ziprecruiter.com","dice.com","greenhouse.io","lever.co","workday.com","bamboohr.com","jobvite.com"]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    def extract(self, job_data: Dict) -> Dict[str, Any]:
        text = self._get_full_text(job_data)
        text_lower = text.lower()
        f: Dict[str, Any] = {}

        f["text_length"] = len(text)
        f["word_count"] = len(text.split())
        f["has_description"] = 1 if len(job_data.get("description", "")) > 50 else 0
        f["has_requirements"] = 1 if len(job_data.get("requirements", "")) > 20 else 0
        f["has_company_name"] = 1 if job_data.get("company", "").strip() else 0
        f["has_salary"] = 1 if job_data.get("salary", "").strip() else 0
        f["has_location"] = 1 if job_data.get("location", "").strip() else 0
        f["has_job_title"] = 1 if job_data.get("title", "").strip() else 0

        f["exclamation_count"] = text.count("!")
        f["caps_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        f["dollar_sign_count"] = text.count("$")
        f["email_count"] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        f["url_count"] = len(re.findall(r'https?://\S+', text))

        f["suspicious_score"] = 0.0
        f["detected_patterns"] = []
        for category, info in SUSPICIOUS_KEYWORDS.items():
            found = [kw for kw in info["keywords"] if kw in text_lower]
            if found:
                f["suspicious_score"] += info["weight"] * len(found)
                f["detected_patterns"].append({
                    "category": info["label"], "keywords": found, "weight": info["weight"],
                    "severity": "HIGH" if info["weight"] >= 0.9 else "MEDIUM" if info["weight"] >= 0.7 else "LOW"
                })

        legit_matches = [kw for kw in LEGITIMATE_INDICATORS if kw in text_lower]
        f["legitimacy_score"] = len(legit_matches)
        f["legit_keywords"] = legit_matches
        f["from_trusted_source"] = 1 if any(d in job_data.get("source_url", "") for d in TRUSTED_DOMAINS) else 0
        f["avg_sentence_length"] = self._avg_sentence_length(text)
        f["has_proper_punctuation"] = 1 if re.search(r'[.!?]', text) else 0
        f["full_text"] = text_lower
        return f

    def _get_full_text(self, job_data: Dict) -> str:
        parts = [job_data.get(k, "") for k in ("title","company","description","requirements","salary","location","full_text")]
        return " ".join(p for p in parts if p)

    def _avg_sentence_length(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return sum(len(s.split()) for s in sentences) / max(len(sentences), 1)


def _to_struct_vec(f: Dict) -> np.ndarray:
    return np.array([
        f["word_count"], f["has_description"], f["has_requirements"],
        f["has_company_name"], f["has_salary"], f["has_location"], f["has_job_title"],
        f["exclamation_count"], f["caps_ratio"], f["dollar_sign_count"],
        f["email_count"], f["url_count"], f["suspicious_score"],
        f["legitimacy_score"], f["from_trusted_source"],
        f["avg_sentence_length"], f["has_proper_punctuation"],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _train_and_save():
    try:
        import scipy.sparse as sp
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn not installed — using rule-based fallback")
        return None

    extractor = FeatureExtractor()
    X_struct, X_text, y = [], [], []
    for job_dict, label in TRAINING_SAMPLES:
        feats = extractor.extract(job_dict)
        X_struct.append(_to_struct_vec(feats))
        X_text.append(feats["full_text"])
        y.append(label)

    X_struct = np.array(X_struct)
    y = np.array(y)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(X_text)

    X_combined = sp.hstack([sp.csr_matrix(X_struct), X_tfidf]).toarray()

    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight="balanced")
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42, solver="lbfgs")
    ensemble = VotingClassifier([("rf", rf), ("lr", lr)], voting="soft", weights=[0.6, 0.4])
    ensemble.fit(X_combined, y)

    bundle = {"ensemble": ensemble, "tfidf": tfidf}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    logger.info("Model trained and saved → %s", MODEL_PATH)
    return bundle


def _load_or_train():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as fh:
                bundle = pickle.load(fh)
            logger.info("Loaded trained model from %s", MODEL_PATH)
            return bundle
        except Exception as e:
            logger.warning("Model load failed (%s) — retraining", e)
    return _train_and_save()


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class ExplainabilityEngine:
    def generate(self, features: Dict, prediction: int, confidence: float, component_scores: Dict) -> Dict:
        reasons, highlighted_keywords, risk_factors = [], [], []

        for p in features.get("detected_patterns", []):
            risk_factors.append({"factor": p["category"], "keywords": p["keywords"],
                                  "severity": p["severity"], "impact": "Increases fraud probability"})
            highlighted_keywords.extend(p["keywords"])
            reasons.append(f"⚠️ {p['category']}: {', '.join(p['keywords'][:3])}")

        if not features["has_company_name"]:
            reasons.append("⚠️ No company name — legitimate employers always identify themselves")
            risk_factors.append({"factor": "Missing Company Name", "severity": "HIGH", "impact": "Anonymity is a fraud indicator"})
        if not features["has_description"]:
            reasons.append("⚠️ Missing job description — vague postings often hide fraudulent intent")
        if not features["has_requirements"]:
            reasons.append("⚠️ No requirements listed — real jobs specify qualifications")
        if features["exclamation_count"] > 5:
            reasons.append(f"⚠️ Excessive exclamation marks ({features['exclamation_count']}) — pressure tactic")
        if features["caps_ratio"] > 0.3:
            reasons.append("⚠️ Excessive CAPITALIZATION — aggressive formatting is a fraud signal")

        legit = features.get("legit_keywords", [])
        if legit:
            reasons.append(f"✅ Legitimate indicators found: {', '.join(legit[:4])}")
        if features.get("from_trusted_source"):
            reasons.append("✅ Posted on a trusted job platform")
        if features["has_company_name"] and features["has_location"] and features["has_description"]:
            reasons.append("✅ Complete structural information provided")

        recommendations = (
            ["Do NOT provide personal or financial information to this employer",
             "Research the company independently before applying",
             "Be cautious of any upfront payment requests",
             "Verify the job posting on official company websites",
             "Report this posting to the job platform if suspicious"]
            if prediction == 1 else
            ["Verify company identity through official channels before applying",
             "Research reviews on Glassdoor or LinkedIn",
             "Follow standard application procedures",
             "Trust your instincts if anything feels off during the process"]
        )

        return {
            "reasons": reasons[:8],
            "risk_factors": risk_factors,
            "highlighted_keywords": list(set(highlighted_keywords)),
            "recommendations": recommendations,
            "component_breakdown": component_scores,
            "legitimate_signals": legit,
            "model_explanation": self._summary(prediction, confidence, features.get("detected_patterns", []))
        }

    def _summary(self, prediction: int, confidence: float, patterns: List) -> str:
        if prediction == 1:
            high = sum(1 for p in patterns if p.get("severity") == "HIGH")
            return (f"The trained ML ensemble detected {len(patterns)} fraud indicators"
                    + (f" including {high} HIGH severity patterns." if high else ".")
                    + f" Fake probability: {confidence*100:.1f}%.")
        return (f"The ML model found predominantly legitimate signals. "
                f"Legitimacy confidence: {(1-confidence)*100:.1f}%.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class FakeJobDetector:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.explainer = ExplainabilityEngine()
        self._bundle = _load_or_train()
        self._use_ml = self._bundle is not None
        logger.info("FakeJobDetector ready (ML=%s)", self._use_ml)

    def analyze(self, job_data: Dict) -> Dict[str, Any]:
        try:
            features = self.extractor.extract(job_data)
            if self._use_ml:
                prediction, confidence, component_scores = self._ml_predict(features)
            else:
                prediction, confidence, component_scores = self._fallback_predict(features)

            explanation = self.explainer.generate(features, prediction, confidence, component_scores)
            risk_level = self._get_risk_level(confidence, prediction)

            return {
                "prediction": "FAKE" if prediction == 1 else "REAL",
                "is_fake": prediction == 1,
                "confidence_score": round(confidence * 100, 1),
                "fake_probability": round(confidence, 4),
                "real_probability": round(1 - confidence, 4),
                "risk_level": risk_level,
                "explanation": explanation,
                "job_data": {k: job_data.get(k, "N/A") for k in ("title","company","location","salary")},
                "features_summary": {
                    "word_count": features["word_count"],
                    "suspicious_patterns_found": len(features.get("detected_patterns", [])),
                    "legitimate_signals_found": features.get("legitimacy_score", 0),
                    "structural_completeness": f"{(features['has_company_name']+features['has_description']+features['has_requirements']+features['has_location'])*25}%"
                },
                "model_type": "Trained ML (RF + LR Ensemble)" if self._use_ml else "Rule-Based Fallback",
                "status": "success"
            }
        except Exception as e:
            logger.error("Detection error: %s", e)
            raise

    def _ml_predict(self, features: Dict) -> Tuple[int, float, Dict]:
        import scipy.sparse as sp
        struct_vec = _to_struct_vec(features).reshape(1, -1)
        tfidf_vec = self._bundle["tfidf"].transform([features["full_text"]])
        X = sp.hstack([sp.csr_matrix(struct_vec), tfidf_vec]).toarray()

        proba = self._bundle["ensemble"].predict_proba(X)[0]
        fake_prob = float(proba[1])
        prediction = 1 if fake_prob > 0.5 else 0

        rf_prob = float(self._bundle["ensemble"].estimators_[0].predict_proba(X)[0][1])
        lr_prob = float(self._bundle["ensemble"].estimators_[1].predict_proba(X)[0][1])

        return prediction, fake_prob, {
            "random_forest": round(rf_prob, 3),
            "logistic_regression": round(lr_prob, 3),
            "ensemble": round(fake_prob, 3),
        }

    def _fallback_predict(self, features: Dict) -> Tuple[int, float, Dict]:
        sus = features.get("suspicious_score", 0)
        legit = features.get("legitimacy_score", 0)
        missing = 4 - (features["has_company_name"] + features["has_description"] +
                       features["has_requirements"] + features["has_location"])
        raw = sus * 0.25 - legit * 0.08 + missing * 0.07
        fake_prob = 1 / (1 + math.exp(-raw))
        return (1 if fake_prob > 0.5 else 0), fake_prob, {"rule_based": round(fake_prob, 3)}

    def _get_risk_level(self, confidence: float, prediction: int) -> Dict[str, str]:
        if prediction == 0:
            return ({"level":"LOW","label":"Low Risk","color":"green","description":"Appears legitimate"}
                    if confidence < 0.3 else
                    {"level":"MEDIUM","label":"Medium Risk","color":"yellow","description":"Some uncertainty — verify before applying"})
        if confidence < 0.65:
            return {"level":"MEDIUM","label":"Medium Risk","color":"orange","description":"Suspicious — verify carefully"}
        if confidence < 0.80:
            return {"level":"HIGH","label":"High Risk","color":"red","description":"Likely fraudulent — do not apply"}
        return {"level":"CRITICAL","label":"Critical Risk","color":"darkred","description":"Almost certainly a scam — avoid completely"}
