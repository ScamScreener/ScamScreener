# 🛡️ ScamScreener

**ScamScreener** is an AI-powered tool designed to help job seekers detect fake job offers and employment frauds. Using a combination of NLP, rule-based matching, and machine learning, it flags suspicious messages and provides a scam risk score along with reasons.

---

## 🚀 Features

- 🧠 **AI-Powered Detection** – NLP and keyword-based analysis of job messages
- 🔍 **Risk Scoring System** – 0–100 scale indicating potential scam risk
- 📬 **Detailed Explanations** – Highlights scam indicators like:
  - Requests for money
  - Free email domains (e.g., Gmail, Yahoo)
  - Unrealistic offers (e.g., “No interview needed”)
- 🏢 **Known Scam Database** – Stores and references flagged companies or domains

---

## 📚 Use Case

This tool is especially useful for:
- Fresh graduates and job seekers
- Career platforms to screen listings
- Cybersecurity education and awareness

---

## 🛠️ Tech Stack

- **Frontend**: HTML/CSS or React (Planned)
- **Backend**: Python + Flask / FastAPI
- **NLP**: spaCy, scikit-learn, transformers (optional)
- **Database**: MongoDB (for logs, known scams)

---

## 📦 Getting Started

```bash
git clone https://github.com/ScamScreener/ScamScreener.git
cd ScamScreener
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
