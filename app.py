from flask import send_from_directory, render_template
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
import datetime

import hashlib
import logging
from collections import Counter
import unicodedata
import sqlite3
from flask import Flask, request, jsonify
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask_cors import CORS
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
ensemble_model = None
tfidf_vectorizer = None
feature_scaler = None
model_stats = {"predictions": 0, "accuracy_rate": 0.0}

# === Auth & Database Configuration ===
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change-this-in-production")
jwt = JWTManager(app)

# --- In-Memory SQLite Setup ---
# Use an in-memory database to avoid Render's ephemeral filesystem
DB_NAME = ":memory:"

@contextmanager
def get_db_conn():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            first_name TEXT,
            last_name TEXT,
            created_at TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            scam_type TEXT,
            company TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS revoked_tokens (
            jti TEXT PRIMARY KEY,
            created_at TEXT
        )''')
        conn.commit()

# Initialize the database when the application starts
with app.app_context():
    init_db()

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload: dict) -> bool:
    jti = jwt_payload.get("jti")
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,))
        row = c.fetchone()
    return row is not None

class AdvancedFeatureExtractor:
    """Advanced feature extraction for fraud detection"""
    
    def __init__(self):
        self.fraud_patterns = {
            'registration_fee': {'patterns': ['registration fee', 'registration cost', 'joining fee'], 'weight': 0.8},
            'processing_fee': {'patterns': ['processing fee', 'processing cost', 'handling fee'], 'weight': 0.8},
            'security_deposit': {'patterns': ['security deposit', 'refundable deposit', 'safety deposit'], 'weight': 0.75},
            'advance_payment': {'patterns': ['advance payment', 'pay advance', 'upfront payment'], 'weight': 0.8},
            'training_fee': {'patterns': ['training fee', 'training cost', 'course fee'], 'weight': 0.6},
            'material_fee': {'patterns': ['material fee', 'kit fee', 'startup kit'], 'weight': 0.65},
            'guaranteed_job': {'patterns': ['guaranteed job', '100% job guarantee', 'job guarantee'], 'weight': 0.7},
            'guaranteed_income': {'patterns': ['guaranteed income', 'guaranteed salary', 'assured income'], 'weight': 0.75},
            'no_experience': {'patterns': ['no experience required', 'no experience needed', 'fresher welcome'], 'weight': 0.4},
            'high_salary': {'patterns': ['earn lakhs', 'high salary', 'big money'], 'weight': 0.6},
            'urgent_hiring': {'patterns': ['urgent hiring', 'immediate joining', 'hurry up'], 'weight': 0.5},
            'limited_seats': {'patterns': ['limited seats', 'few seats left', 'only today'], 'weight': 0.45},
            'act_fast': {'patterns': ['act fast', 'call now', 'apply immediately'], 'weight': 0.4},
            'copy_paste': {'patterns': ['copy paste work', 'copy paste job'], 'weight': 0.6},
            'data_entry': {'patterns': ['simple data entry', 'easy data entry'], 'weight': 0.3},
            'form_filling': {'patterns': ['form filling work', 'form submission'], 'weight': 0.35},
            'survey_work': {'patterns': ['survey work', 'online survey', 'paid survey'], 'weight': 0.3},
            'ad_posting': {'patterns': ['ad posting', 'advertisement posting'], 'weight': 0.4},
            'whatsapp_contact': {'patterns': ['whatsapp me', 'msg on whatsapp', 'whatsapp only'], 'weight': 0.4},
            'telegram_contact': {'patterns': ['telegram me', 'contact telegram'], 'weight': 0.45},
            'direct_contact': {'patterns': ['direct contact', 'personal contact'], 'weight': 0.3},
        }
        self.legitimate_patterns = {
            'interview_process': {'patterns': ['interview required', 'interview process', 'multiple rounds'], 'weight': 0.3},
            'experience_required': {'patterns': ['experience required', 'minimum experience', 'years experience'], 'weight': 0.25},
            'qualifications': {'patterns': ['degree required', 'certification required', 'qualification'], 'weight': 0.2},
            'company_benefits': {'patterns': ['health insurance', 'provident fund', 'medical benefits'], 'weight': 0.25},
            'professional_terms': {'patterns': ['career growth', 'professional development', 'appraisal'], 'weight': 0.15},
            'background_check': {'patterns': ['background verification', 'document verification'], 'weight': 0.3},
        }
        self.suspicious_keywords = [
            'earn', 'money', 'income', 'salary', 'payment', 'fee', 'deposit', 'cost',
            'guaranteed', 'assured', 'confirmed', 'promise', 'urgent', 'immediate',
            'easy', 'simple', 'part time', 'full time', 'home', 'mobile', 'online'
        ]
    
    def extract_numerical_features(self, text):
        features = {}
        salary_patterns = [
            r'₹\s*(\d{1,2}(?:,\d{2})*(?:,\d{3})*)',
            r'(\d{1,2}(?:,\d{2})*(?:,\d{3})*)\s*(?:rupees|rs)',
            r'earn\s*(?:up\s*to\s*)?₹?\s*(\d{1,2}(?:,\d{2})*(?:,\d{3})*)',
            r'salary\s*(?:up\s*to\s*)?₹?\s*(\d{1,2}(?:,\d{2})*(?:,\d{3})*)',
            r'(\d{1,2})\s*lakh',
            r'(\d{1,2})\s*k',
        ]
        salary_amounts = []
        for pattern in salary_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                match_clean = re.sub(r'[,\s]', '', str(match))
                if match_clean.isdigit():
                    salary_amounts.append(int(match_clean))
        if 'lakh' in text.lower():
            lakh_matches = re.findall(r'(\d+(?:\.\d+)?)\s*lakh', text.lower())
            for match in lakh_matches:
                salary_amounts.append(int(float(match) * 100000))
        if re.search(r'\d+k\b', text.lower()):
            k_matches = re.findall(r'(\d+)k\b', text.lower())
            for match in k_matches:
                salary_amounts.append(int(match) * 1000)
        features['max_salary'] = int(max(salary_amounts)) if salary_amounts else 0
        features['min_salary'] = int(min(salary_amounts)) if salary_amounts else 0
        features['salary_count'] = int(len(salary_amounts))
        features['unrealistic_salary'] = int(1 if features['max_salary'] > 100000 else 0)
        phone_patterns = [
            r'\+91[6-9]\d{9}',
            r'\b[6-9]\d{9}\b',
            r'\b\d{10}\b'
        ]
        phone_count = 0
        for pattern in phone_patterns:
            phone_count += len(re.findall(pattern, text))
        features['phone_numbers'] = int(phone_count)
        features['multiple_phones'] = int(1 if phone_count > 1 else 0)
        features['text_length'] = int(len(text))
        features['word_count'] = int(len(text.split()))
        features['exclamation_count'] = int(text.count('!'))
        features['caps_ratio'] = float(sum(1 for c in text if c.isupper()) / max(len(text), 1))
        features['digit_ratio'] = float(sum(1 for c in text if c.isdigit()) / max(len(text), 1))
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        emojis = emoji_pattern.findall(text)
        features['emoji_count'] = int(len(emojis))
        fraud_emojis = ['🚀', '💰', '⭐', '🔥', '💼', '💯', '🎯', '💎', '🌟', '✨', '🤑', '💵', '🏆', '⚡']
        fraud_emoji_count = sum(text.count(emoji) for emoji in fraud_emojis)
        features['fraud_emoji_count'] = int(fraud_emoji_count)
        return features
    
    def extract_linguistic_features(self, text):
        features = {}
        words = text.split()
        features['avg_word_length'] = float(np.mean([len(word) for word in words]) if words else 0)
        features['unique_word_ratio'] = float(len(set(words)) / max(len(words), 1))
        suspicious_count = sum(1 for word in words if word.lower() in self.suspicious_keywords)
        features['suspicious_keyword_density'] = float(suspicious_count / max(len(words), 1))
        word_counts = Counter(word.lower() for word in words)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 1
        features['max_word_repetition'] = int(most_common_count)
        features['question_count'] = int(text.count('?'))
        features['uncertainty_words'] = int(sum(1 for word in ['maybe', 'might', 'could', 'probably'] if word in text.lower()))
        return features
    
    def calculate_pattern_score(self, text):
        text_lower = text.lower()
        fraud_score = 0
        legitimate_score = 0
        indicators = []
        for category, pattern_info in self.fraud_patterns.items():
            for pattern in pattern_info['patterns']:
                if pattern in text_lower:
                    fraud_score += pattern_info['weight']
                    indicators.append(f"{category.replace('_', ' ').title()}: {pattern}")
                    break
        for category, pattern_info in self.legitimate_patterns.items():
            for pattern in pattern_info['patterns']:
                if pattern in text_lower:
                    legitimate_score += pattern_info['weight']
                    indicators.append(f"Legitimate: {category.replace('_', ' ').title()}")
                    break
        final_score = fraud_score - legitimate_score
        final_score = max(0, min(final_score, 1.0))
        return final_score, indicators
    
    def extract_all_features(self, text):
        numerical_features = self.extract_numerical_features(text)
        linguistic_features = self.extract_linguistic_features(text)
        pattern_score, indicators = self.calculate_pattern_score(text)
        text_lower = text.lower()
        for category, pattern_info in self.legitimate_patterns.items():
            feature_name = f"legitimate_{category}"
            feature_value = 0
            for pattern in pattern_info['patterns']:
                if pattern in text_lower:
                    feature_value = 1
                    break
            numerical_features[feature_name] = int(feature_value)
        all_features = {
            **numerical_features,
            **linguistic_features,
            'pattern_score': float(pattern_score),
            'indicator_count': int(len(indicators))
        }
        return all_features, indicators

class AdvancedPreprocessor:
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 
            'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'you', 
            'your', 'we', 'our', 'this', 'have', 'had', 'or', 'but', 'can', 'do', 'not', 'if', 
            'up', 'so', 'they', 'them', 'their', 'what', 'who', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'now'
        }
    
    def normalize_text(self, text):
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        replacements = {
            'whatsapp': 'whatsapp',
            'wat\'sapp': 'whatsapp', 
            'watsapp': 'whatsapp',
            'guaranted': 'guaranteed',
            'guarenteed': 'guaranteed',
            'recieve': 'receive',
            'seperate': 'separate',
            'oppurtunity': 'opportunity',
            'experiance': 'experience',
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        text = re.sub(r'rs\.?\s*', '₹', text)
        text = re.sub(r'rupees?\s*', '₹', text)
        text = re.sub(r'\+91\s*', '', text)
        text = re.sub(r'(\d{5})\s*(\d{5})', r'\1\2', text)
        text = re.sub(r'[!]{3,}', '!!!', text)
        text = re.sub(r'[?]{3,}', '???', text)
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[^\w\s₹!?.,()-]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def tokenize_and_filter(self, text):
        text = self.normalize_text(text)
        words = text.split()
        filtered_words = [
            word for word in words 
            if word not in self.stopwords and len(word) > 2 and not word.isdigit()
        ]
        return ' '.join(filtered_words)

class EnhancedFraudDetector:
    """Main fraud detection class with ensemble learning"""
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.preprocessor = AdvancedPreprocessor()
        self.ensemble_model = None
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.feature_names = []
        
    def create_training_data(self, dataset_path=None):
        if dataset_path and os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                legitimate_jobs = data['legitimate']
                fraudulent_jobs = data['fraudulent']
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load dataset from {dataset_path}: {e}. Falling back to default data.")
                legitimate_jobs, fraudulent_jobs = self._get_default_data()
        else:
            legitimate_jobs, fraudulent_jobs = self._get_default_data()
        return legitimate_jobs, fraudulent_jobs
    
    def _get_default_data(self):
        legitimate_jobs = [
            "Software Engineer position at leading technology company. Requirements: Bachelor's degree in Computer Science, 3+ years Python/Java experience. Competitive salary, health insurance, and career growth opportunities. Interview process includes technical rounds.",
            "Marketing Manager role in Mumbai. MBA in Marketing required with 4+ years experience in digital marketing. Handle campaign management and team leadership. Salary range: ₹8-12 lakhs. Standard benefits provided.",
            "Data Analyst position at analytics firm. Must know SQL, Python, R, and Excel. Fresh graduates with strong analytical skills welcome. Training provided. Good growth opportunities in data science.",
            "Senior Accountant needed for manufacturing company. CA/CMA qualification required with 5+ years experience. Handle financial reporting, taxation, and compliance. Competitive package with medical benefits.",
            "Customer Support Representative for telecom company. Good communication skills and graduation required. Rotational shifts. Training provided. Standard salary and incentives.",
            "Business Development Executive for B2B sales. MBA preferred with 2+ years experience. Field work involved. Target-based incentives. Company vehicle provided.",
            "HR Manager position in healthcare sector. MBA HR with 6+ years experience in recruitment and employee relations. Handle policy implementation. Competitive package.",
            "Backend Developer role at fintech startup. Experience in Node.js, Python, databases required. Equity options available. Modern work environment. Technical interview process.",
            "Content Writer for digital marketing agency. English literature background preferred. Create blogs, articles, and marketing content. Portfolio review required. Creative work environment.",
            "Quality Assurance Engineer in automotive industry. Engineering degree with knowledge of testing methodologies. Good analytical skills. Manufacturing experience preferred.",
            "Finance Manager for retail chain. CA with 7+ years experience in financial planning and analysis. Multiple location handling. Competitive salary with performance bonus.",
            "UI/UX Designer position. Design degree or equivalent experience. Portfolio of web and mobile designs required. Collaborative work environment. Modern design tools used.",
            "Operations Manager for logistics company. Engineering or MBA with supply chain experience. Team management skills required. Performance-based compensation.",
            "Research Analyst for investment firm. CFA or MBA Finance preferred. Strong analytical and communication skills. Financial modeling experience required.",
            "Product Manager role at e-commerce company. Technical background with 4+ years product management experience. Cross-functional collaboration required. Stock options available.",
        ] * 10
        fraudulent_jobs = [
            "🚀 URGENT HIRING! Earn ₹50,000 monthly from home! Simple data entry work. No experience needed! Just pay ₹3,000 registration fee. 100% genuine company! Call 9876543210 WhatsApp me now!",
            "💰 Amazing opportunity! Work from mobile phone! Copy-paste work! Earn ₹40,000 monthly! Pay only ₹2,500 security deposit. Money back guarantee! Limited seats! Apply today! WhatsApp 8765432109",
            "🔥 Government approved company! Form filling job! Earn ₹35,000 monthly! No qualifications required! Training fee ₹2,000 only! 100% job guarantee! Immediate joining! Call now 7654321098",
            "⭐ Part-time work from home! Survey work on mobile! Earn ₹25,000 monthly! Work 2 hours daily! Registration fee ₹1,500! Refundable deposit! Join today! Limited time offer!",
            "💎 Online typing work! Easy copy-paste job! Earn ₹45,000 monthly! No experience needed! Processing fee ₹3,500! Guaranteed income! Work from anywhere! Hurry up! Few seats left!",
            "🌟 Ad posting job! Social media work! Earn ₹30,000 monthly! Mobile/laptop work! Training provided! Registration ₹2,200! 100% genuine! Immediate start! WhatsApp only 9988776655",
            "💯 Data entry work! Government project! Earn ₹55,000 monthly! 10th pass eligible! Security deposit ₹4,000! Refundable! Assured job! Call immediately! Don't miss this chance!",
            "🎯 Form submission job! Work from home! Earn ₹38,000 monthly! Simple work! No experience! Material fee ₹2,800! Money back guarantee! Limited vacancy! Apply now!",
            "🚀 Online survey work! Earn ₹42,000 monthly! Work on mobile! Part-time job! Registration fee ₹3,200! Genuine company! 100% payment guarantee! Call today 8877665544",
            "💰 Copy-paste work from home! Earn ₹48,000 monthly! Easy job! No qualifications! Training fee ₹2,700! Refundable! Immediate joining! WhatsApp me for details!",
            "🔥 Data typing job! Government approved! Earn ₹52,000 monthly! Simple English typing! Registration ₹3,800! Job guarantee! Work from home! Call now!",
            "⭐ Online form filling! Earn ₹33,000 monthly! Mobile work! Part-time! Processing fee ₹2,100! Money back guarantee! Join today! Limited seats available!",
            "💎 Advertisement posting work! Social media job! Earn ₹41,000 monthly! No experience! Training provided! Fee ₹2,900! Guaranteed payment! Apply immediately!",
            "🌟 Survey completion job! Work from mobile! Earn ₹29,000 monthly! Easy work! Registration ₹1,800! Refundable deposit! 100% genuine! Call now for details!",
            "💯 Simple data entry! Government project! Earn ₹46,000 monthly! 12th pass eligible! Security fee ₹3,300! Job assured! Work from home! Don't wait!",
        ] * 8
        return legitimate_jobs, fraudulent_jobs
    
    def prepare_features(self, texts, labels):
        feature_matrix = []
        text_features = []
        for text in texts:
            features, _ = self.feature_extractor.extract_all_features(text)
            feature_matrix.append(list(features.values()))
            processed_text = self.preprocessor.tokenize_and_filter(text)
            text_features.append(processed_text)
        sample_features, _ = self.feature_extractor.extract_all_features(texts[0])
        self.feature_names = list(sample_features.keys())
        feature_matrix = np.array(feature_matrix)
        self.feature_scaler = MinMaxScaler()
        feature_matrix_scaled = self.feature_scaler.fit_transform(feature_matrix)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words=None
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(text_features)
        combined_features = np.hstack([feature_matrix_scaled, tfidf_features.toarray()])
        return combined_features, np.array(labels)
    
    def train_model(self):
        logger.info("Creating training dataset...")
        legitimate_jobs, fraudulent_jobs = self.create_training_data()
        all_texts = legitimate_jobs + fraudulent_jobs
        all_labels = [0] * len(legitimate_jobs) + [1] * len(fraudulent_jobs)
        logger.info(f"Training on {len(all_texts)} samples ({len(legitimate_jobs)} legitimate, {len(fraudulent_jobs)} fraudulent)")
        X, y = self.prepare_features(all_texts, all_labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("Training ensemble model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        nb_model = MultinomialNB(alpha=1.0)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('lr', lr_model),
                ('nb', nb_model)
            ],
            voting='soft'
        )
        self.ensemble_model.fit(X_train, y_train)
        train_score = self.ensemble_model.score(X_train, y_train)
        test_score = self.ensemble_model.score(X_test, y_test)
        logger.info(f"Training accuracy: {train_score:.3f}")
        logger.info(f"Testing accuracy: {test_score:.3f}")
        model_stats["accuracy_rate"] = test_score
        self.save_models()
        return test_score
    
    def save_models(self):
        try:
            with open('enhanced_ensemble_model.pkl', 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            with open('enhanced_tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            with open('enhanced_feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            with open('feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            logger.info("✅ Models saved successfully")
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    def load_models(self):
        try:
            if all(os.path.exists(f) for f in ['enhanced_ensemble_model.pkl', 'enhanced_tfidf_vectorizer.pkl', 'enhanced_feature_scaler.pkl']):
                with open('enhanced_ensemble_model.pkl', 'rb') as f:
                    self.ensemble_model = pickle.load(f)
                with open('enhanced_tfidf_vectorizer.pkl', 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                with open('enhanced_feature_scaler.pkl', 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                with open('feature_names.pkl', 'rb') as f:
                    self.feature_names = pickle.load(f)
                logger.info("✅ Enhanced models loaded successfully")
                return True
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
        logger.info("🔄 Training new model...")
        self.train_model()
        return True
    
    def predict(self, text):
        try:
            features, indicators = self.feature_extractor.extract_all_features(text)
            feature_vector = np.array([list(features.values())])
            feature_vector_scaled = self.feature_scaler.transform(feature_vector)
            processed_text = self.preprocessor.tokenize_and_filter(text)
            tfidf_features = self.tfidf_vectorizer.transform([processed_text])
            combined_features = np.hstack([feature_vector_scaled, tfidf_features.toarray()])
            fraud_probability = self.ensemble_model.predict_proba(combined_features)[0][1]
            pattern_score, pattern_indicators = self.feature_extractor.calculate_pattern_score(text)
            final_score = (fraud_probability * 0.7) + (pattern_score * 0.3)
            all_indicators = list(set(indicators + pattern_indicators))
            model_stats["predictions"] += 1
            return {
                'fraud_probability': float(final_score),
                'ml_score': float(fraud_probability),
                'pattern_score': float(pattern_score),
                'indicators': all_indicators,
                'features': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in features.items()}
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            pattern_score, indicators = self.feature_extractor.calculate_pattern_score(text)
            return {
                'fraud_probability': float(pattern_score),
                'ml_score': 0.5,
                'pattern_score': float(pattern_score),
                'indicators': indicators,
                'features': {}
            }

detector = EnhancedFraudDetector()

def get_risk_assessment(fraud_score, indicators):
    indicator_count = len(indicators)
    if fraud_score >= 0.9:
        risk_level = "Critical"
        recommendation = "🚨 CRITICAL RISK - This is almost certainly a scam. Avoid completely!"
        confidence = "Very High"
        action = "Block immediately and report to authorities"
    elif fraud_score >= 0.75:
        risk_level = "Very High"
        recommendation = "🚨 VERY HIGH RISK - Strong indicators of fraud. Do not proceed!"
        confidence = "Very High"
        action = "Avoid completely, warn others"
    elif fraud_score >= 0.6:
        risk_level = "High"
        recommendation = "🚨 HIGH RISK - Multiple fraud indicators detected. Exercise extreme caution!"
        confidence = "High" 
        action = "Thoroughly verify company and job details before proceeding"
    elif fraud_score >= 0.4:
        risk_level = "Medium"
        recommendation = "⚠️ MEDIUM RISK - Several concerning patterns found. Investigate carefully!"
        confidence = "Medium"
        action = "Research company, verify contact details, ask detailed questions"
    elif fraud_score >= 0.25:
        risk_level = "Low-Medium"
        recommendation = "⚠️ LOW-MEDIUM RISK - Some red flags present. Proceed with caution!"
        confidence = "Medium"
        action = "Verify company credentials and job details"
    elif fraud_score >= 0.1:
        risk_level = "Low"
        recommendation = "⚠️ LOW RISK - Minor concerns detected. Basic verification recommended!"
        confidence = "Medium"
        action = "Basic company verification recommended"
    else:
        risk_level = "Very Low"
        recommendation = "✅ VERY LOW RISK - Appears legitimate based on analysis."
        confidence = "High"
        action = "Standard due diligence recommended"
    if indicator_count >= 8:
        confidence = "Very High"
    elif indicator_count >= 5:
        confidence = "High"
    elif indicator_count >= 3:
        confidence = "Medium"
    elif indicator_count <= 1:
        confidence = "Low"
    return {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "confidence": confidence,
        "action": action,
        "indicator_count": indicator_count,
        "severity_score": min(int(fraud_score * 10), 10)
    }

def analyze_job_posting(text):
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "text_hash": hashlib.md5(text.encode()).hexdigest()[:10],
        "text_length": len(text),
        "word_count": len(text.split())
    }
    prediction_result = detector.predict(text)
    risk_info = get_risk_assessment(
        prediction_result['fraud_probability'], 
        prediction_result['indicators']
    )
    analysis.update({
        "fraud_probability": round(float(prediction_result['fraud_probability']), 4),
        "ml_score": round(float(prediction_result['ml_score']), 4),
        "pattern_score": round(float(prediction_result['pattern_score']), 4),
        "is_fraudulent": bool(prediction_result['fraud_probability'] > 0.5),
        "fraud_indicators": prediction_result['indicators'],
        "risk_assessment": risk_info,
        "features": prediction_result['features']
    })
    financial_indicators = [ind for ind in prediction_result['indicators'] 
                          if any(word in ind.lower() for word in ['fee', 'deposit', 'payment', 'cost'])]
    promise_indicators = [ind for ind in prediction_result['indicators']
                         if any(word in ind.lower() for word in ['guaranteed', 'assured', 'promise'])]
    urgency_indicators = [ind for ind in prediction_result['indicators']
                         if any(word in ind.lower() for word in ['urgent', 'immediate', 'hurry', 'limited'])]
    analysis["indicator_categories"] = {
        "financial_red_flags": financial_indicators,
        "unrealistic_promises": promise_indicators,
        "urgency_tactics": urgency_indicators,
        "other_indicators": [ind for ind in prediction_result['indicators'] 
                           if ind not in financial_indicators + promise_indicators + urgency_indicators]
    }
    return analysis

logger.info("🚀 Initializing Enhanced Fraud Detection System...")
detector.load_models()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = (data.get("message") or data.get("text") or "").strip()
        if not message:
            return jsonify({"error": "Empty message provided"}), 400
        analysis = analyze_job_posting(message)
        result = {
            "is_scam": bool(analysis["is_fraudulent"]),
            "confidence": float(analysis["fraud_probability"]),
            "fraud_indicators": analysis["fraud_indicators"],
            "risk_level": analysis["risk_assessment"]["risk_level"],
            "recommendation": analysis["risk_assessment"]["recommendation"], 
            "detection_confidence": analysis["risk_assessment"]["confidence"],
            "pattern_score": float(analysis["pattern_score"]),
            "ml_score": float(analysis["ml_score"]),
            "indicator_count": int(analysis["risk_assessment"]["indicator_count"]),
            "severity_score": int(analysis["risk_assessment"]["severity_score"]),
            "suggested_action": analysis["risk_assessment"]["action"]
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        message = (data.get("message") or data.get("text") or "").strip()
        if not message:
            return jsonify({"error": "Empty message provided"}), 400
        analysis = analyze_job_posting(message)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

@app.route("/retrain", methods=["POST"])
def retrain_model():
    try:
        logger.info("🔄 Retraining model...")
        accuracy = detector.train_model()
        return jsonify({
            "message": "✅ Model retrained successfully",
            "new_accuracy": f"{accuracy:.1%}",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Retraining error: {e}")
        return jsonify({"error": "Retraining failed", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "version": "2.0 Enhanced",
        "model_loaded": detector.ensemble_model is not None,
        "vectorizer_loaded": detector.tfidf_vectorizer is not None,
        "scaler_loaded": detector.feature_scaler is not None,
        "features_count": len(detector.feature_names) if detector.feature_names else 0,
        "accuracy": f"{model_stats.get('accuracy_rate', 0):.1%}",
        "total_predictions": model_stats.get('predictions', 0),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "model_stats": model_stats,
        "feature_count": len(detector.feature_names) if detector.feature_names else 0,
        "supported_languages": ["English", "Hindi (transliterated)"],
        "detection_categories": [
            "Financial Fraud Indicators",
            "Unrealistic Job Promises", 
            "Urgency Tactics",
            "Communication Red Flags",
            "Salary Analysis",
            "Text Pattern Analysis"
        ]
    })

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Please provide a list of messages"}), 400
        if len(messages) > 100:
            return jsonify({"error": "Maximum 100 messages allowed per batch"}), 400
        results = []
        for i, message in enumerate(messages):
            try:
                analysis = analyze_job_posting(str(message).strip())
                results.append({
                    "index": i,
                    "message": message[:100] + "..." if len(message) > 100 else message,
                    "is_fraudulent": analysis["is_fraudulent"],
                    "fraud_probability": analysis["fraud_probability"],
                    "risk_level": analysis["risk_assessment"]["risk_level"],
                    "indicator_count": analysis["risk_assessment"]["indicator_count"]
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "message": message[:100] + "..." if len(message) > 100 else message,
                    "error": str(e)
                })
        valid_results = [r for r in results if "error" not in r]
        fraud_count = sum(1 for r in valid_results if r["is_fraudulent"])
        summary = {
            "total_messages": len(messages),
            "processed_successfully": len(valid_results),
            "fraud_detected": fraud_count,
            "legitimate_detected": len(valid_results) - fraud_count,
            "fraud_rate": fraud_count / len(valid_results) if valid_results else 0
        }
        return jsonify({
            "results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        return jsonify({"error": "Batch prediction failed", "details": str(e)}), 500

# === Authentication Endpoints ===
@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    if not username or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400
    hashed_password = generate_password_hash(password)
    created_at = datetime.utcnow().isoformat()
    with get_db_conn() as conn:
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (username, email, password, first_name, last_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (username, email, hashed_password, first_name, last_name, created_at),
            )
            conn.commit()
            user_id = c.lastrowid
            access_token = create_access_token(identity=str(user_id))
            refresh_token = create_refresh_token(identity=str(user_id))
            return jsonify({
                "msg": "User created successfully",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user_id,
                    "username": username,
                    "email": email
                }
            }), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": "Username or email already exists"}), 409

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    username_or_email = data.get("username_or_email")
    password = data.get("password")
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? OR email=?", (username_or_email, username_or_email))
        user = c.fetchone()
    if user and check_password_hash(user['password'], password):
        user_id = user['id']
        access_token = create_access_token(identity=str(user_id))
        refresh_token = create_refresh_token(identity=str(user_id))
        return jsonify({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": user_id,
                "username": user['username'],
                "email": user['email']
            }
        }), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/dashboard/stats", methods=["GET"])
@jwt_required()
def dashboard_stats():
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM reports")
        total_reports = c.fetchone()[0]
    return jsonify({"total_users": total_users, "total_reports": total_reports})

@app.route("/dashboard/reports", methods=["GET"])
@jwt_required()
def dashboard_reports():
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, message, scam_type, company, created_at FROM reports ORDER BY id DESC LIMIT 10")
        rows = c.fetchall()
    reports = [
        {"id": r[0], "message": r[1], "scam_type": r[2], "company": r[3], "created_at": r[4]}
        for r in rows
    ]
    return jsonify(reports)

@app.route("/auth/logout", methods=["POST"])
@jwt_required()
def logout():
    jwt_payload = get_jwt()
    jti = jwt_payload.get("jti")
    created_at = datetime.utcnow().isoformat()
    with get_db_conn() as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT OR IGNORE INTO revoked_tokens (jti, created_at) VALUES (?, ?)", (jti, created_at))
            conn.commit()
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
    return jsonify({"message": "Logged out"}), 200

@app.route("/auth/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh_access():
    current_user = get_jwt_identity()
    new_token = create_access_token(identity=current_user)
    return jsonify({"access_token": new_token})

@app.route("/report", methods=["POST"])
@jwt_required(refresh=False)
def submit_report():
    try:
        user_id = get_jwt_identity()
        data = request.get_json(force=True)
        message = data.get("message")
        scam_type = data.get("scam_type")
        company = data.get("company")
        if not message or not scam_type:
            return jsonify({"error": "Message and scam_type are required"}), 400
        with get_db_conn() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO reports (message, scam_type, company, created_at, user_id)
                VALUES (?, ?, ?, ?, ?)
            """, (message, scam_type, company, datetime.now().isoformat(), user_id))
            conn.commit()
            report_id = c.lastrowid
            return jsonify({
                "message": "Report submitted successfully",
                "report": {
                    "id": report_id,
                    "user_id": user_id,
                    "message": message,
                    "scam_type": scam_type,
                    "company": company,
                    "created_at": datetime.now().isoformat()
                }
            }), 201
    except Exception as e:
        print("❌ Report error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/reports", methods=["GET"])
def list_reports():
    try:
        with get_db_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT r.id, r.message, r.scam_type, r.company, r.created_at, r.user_id, u.username FROM reports r LEFT JOIN users u ON r.user_id = u.id ORDER BY r.created_at DESC")
            rows = c.fetchall()
        items = []
        for r in rows:
            items.append({
                "id": r["id"],
                "message": r["message"],
                "scam_type": r["scam_type"],
                "company": r["company"],
                "created_at": r["created_at"],
                "user_id": r["user_id"],
                "username": r["username"]
            })
        return jsonify(items), 200
    except Exception as e:
        logger.error(f"Reports fetch error: {e}")
        return jsonify({"error": "reports_failed", "details": str(e)}), 500

@app.route("/db_stats", methods=["GET"])
def db_stats():
    try:
        with get_db_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) AS c FROM users")
            total_users = c.fetchone()["c"]
            c.execute("SELECT COUNT(*) AS c FROM reports")
            total_reports = c.fetchone()["c"]
        return jsonify({"total_users": total_users, "total_reports": total_reports}), 200
    except Exception as e:
        logger.error(f"DB stats error: {e}")
        return jsonify({"error": "db_stats_failed", "details": str(e)}), 500

# ==========================
# Serve Frontend HTML pages
# ==========================
@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/signup")
def signup_page():
    return render_template("signup.html")

@app.route("/dashboard")
@jwt_required(optional=True)
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/howitworks")
def howitworks_page():
    return render_template("howitworks.html")

@app.route("/pricing")
def pricing_page():
    return render_template("pricing.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/contact")
def contact_page():
    return render_template("contact.html")

if __name__ == "__main__":
    print("🎯 Enhanced Fraud Detection API v2.0 Ready!")
    print("📈 Features:")
    print("   • Ensemble Machine Learning (Random Forest + Logistic Regression + Naive Bayes)")
    print("   • Advanced Feature Extraction (60+ features)")
    print("   • Pattern Recognition (50+ fraud patterns)")
    print("   • Real-time Risk Assessment")
    print("   • Batch Processing Support")
    print("   • Comprehensive Analytics")
    print("🌐 Endpoints:")
    print("   • POST /predict - Simple fraud prediction")
    print("   • POST /analyze - Comprehensive analysis")
    print("   • POST /batch_predict - Batch processing")
    print("   • POST /retrain - Model retraining")
    print("   • GET /health - Health check")
    print("   • GET /stats - Model statistics")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
