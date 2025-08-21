# 🛡️ ScamScreener

**ScamScreener** is an AI-powered tool designed to help job seekers detect fake job offers and employment frauds. Using a custom ensemble machine learning model, the application analyzes job offer messages, provides a risk score, and highlights potential scam indicators.

---

## 🚀 Features

* **AI-Powered Detection**: A custom-trained machine learning model analyzes job offers using a multi-faceted approach, combining rule-based checks, natural language processing (NLP), and pattern recognition.
* **Risk Scoring System**: Provides an immediate scam risk score (0-100 scale) to indicate the potential for fraud.
* **Detailed Analysis**: The application highlights specific scam indicators found in the text and offers clear recommendations and suggested actions for the user.
* **User Authentication**: Includes a complete user authentication flow with signup and login pages, and a secure backend using JWT for session management.
* **Community Reporting**: Allows authenticated users to report new scams, contributing to a community-driven database that can be used to retrain the model.

---

## 🛠️ Tech Stack

* **Backend**: Python, Flask, `scikit-learn`.
* **Frontend**: HTML, CSS, and vanilla JavaScript.
* **Database**: SQLite.
* **Dependencies**: The project relies on the libraries listed in `requirements.txt`.

---

## 📦 Getting Started

To get a copy of the project up and running on your local machine, follow these simple steps.

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/ScamScreener/ScamScreener.git](https://github.com/ScamScreener/ScamScreener.git)
    cd ScamScreener
    ```

2.  **Set up the Python environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application**:
    The first time you run the application, it will automatically train the machine learning model and save the necessary files. This process may take a few moments.
    ```bash
    python app.py
    ```

5.  **Access the application**:
    Open your web browser and navigate to `http://127.0.0.1:5000` to start using ScamScreener.

---
