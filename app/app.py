import streamlit as st
import joblib
import os
import numpy as np
import shap

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Fake News Intelligence System",
    layout="wide"
)

# ---------------- LOAD CSS ----------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- THEME STATE ----------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

theme_toggle = st.sidebar.toggle("🌗 Dark Mode", value=True)
st.session_state["theme"] = "dark" if theme_toggle else "light"

# Inject theme flag
st.markdown(
    f'<div id="theme-root" data-theme="{st.session_state["theme"]}"></div>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "models", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl"))

# ---------------- SHAP ----------------
explainer = shap.LinearExplainer(model, vectorizer.transform(["sample text"]))

# ---------------- SESSION ----------------
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# =========================================================
# 🧠 HEADER
# =========================================================
st.markdown('<div class="title">🧠 Fake News Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">End-to-End NLP + Machine Learning System for Misinformation Detection</div>', unsafe_allow_html=True)

# =========================================================
# 🚀 HERO SECTION
# =========================================================
st.markdown("""
## 📰 Project Overview

This system leverages **Natural Language Processing (NLP)** and **Machine Learning**  
to automatically classify news articles as:

- ✅ **REAL NEWS**
- 🚨 **FAKE NEWS**

### 🎯 Objective
Combat misinformation using AI by analyzing linguistic patterns, semantics, and statistical features.

### ⚙️ Core Capabilities
- Real-time text classification  
- Explainable AI (SHAP)  
- High accuracy (~99%)  
- Lightweight and fast inference  
""")

# =========================================================
# 📊 TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔍 Detector", "📊 Model Insights", "📘 Project Details"])

# =========================================================
# 🔍 DETECTOR
# =========================================================
with tab1:

    st.markdown("### 🧪 Test the Model")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Example (Likely Fake)"):
            st.session_state["input_text"] = "Government announces new economic reforms to boost growth"

    with col2:
        if st.button("Load Example (Likely Real)"):
            st.session_state["input_text"] = "Aliens sign trade agreement with world leaders"

    # st.markdown('<div class="glass">', unsafe_allow_html=True)

    user_input = st.text_area(
        "📝 Enter News Content",
        value=st.session_state["input_text"],
        height=220
    )

    analyze = st.button("🔍 Analyze News")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PREDICTION ----------------
    if analyze:

        if user_input.strip() == "":
            st.warning("⚠️ Please enter valid news content.")
        else:
            vectorized = vectorizer.transform([user_input])
            pred = model.predict(vectorized)[0]
            probs = model.predict_proba(vectorized)[0]
            confidence = probs[pred]

            st.subheader("📊 Prediction Result")

            if pred == 1:
                st.markdown(f"""
                <div class="result-real">
                <h3>✅ REAL NEWS</h3>
                <p>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fake">
                <h3>🚨 FAKE NEWS</h3>
                <p>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

            st.progress(float(confidence))

            # ---------------- FEATURE IMPORTANCE ----------------
            st.subheader("🔍 Key Influencing Words")

            feature_names = np.array(vectorizer.get_feature_names_out())
            top_indices = np.argsort(vectorized.toarray()[0])[-10:]
            top_words = feature_names[top_indices]

            st.info(f"Important words influencing prediction: {', '.join(top_words)}")

            # ---------------- SHAP ----------------
            st.subheader("🧠 Explainable AI (Model Reasoning)")

            shap_values = explainer.shap_values(vectorized)[0]

            top_positive = np.argsort(shap_values)[-5:]
            top_negative = np.argsort(shap_values)[:5]

            st.success("Words pushing prediction towards REAL:")
            st.write(", ".join(feature_names[top_positive]))

            st.error("Words pushing prediction towards FAKE:")
            st.write(", ".join(feature_names[top_negative]))

# =========================================================
# 📊 MODEL INSIGHTS
# =========================================================
with tab2:

    st.markdown("## 📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "99%")
    col2.metric("Precision", "0.99")
    col3.metric("Recall", "0.99")

    st.markdown("### 🧠 Algorithms Used")

    st.write("""
- **Logistic Regression**
  - Best performing model
  - Interpretable and efficient

- **Naive Bayes**
  - Fast baseline model

- **Random Forest**
  - High accuracy but prone to overfitting in this dataset
    """)

    st.markdown("### 📉 Limitations")

    st.warning("""
- Cannot verify real-world facts (only patterns)
- Sensitive to dataset bias
- Works best with full article text
    """)

# =========================================================
# 📘 PROJECT DETAILS
# =========================================================
with tab3:

    st.markdown("## 📘 End-to-End Pipeline")

    st.write("""
### 🔄 Workflow

1. **Data Collection**
   - Real & Fake news datasets

2. **Text Preprocessing**
   - Lowercasing
   - Stopword removal
   - Lemmatization

3. **Feature Engineering**
   - TF-IDF Vectorization

4. **Model Training**
   - Logistic Regression (final)

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score

6. **Deployment**
   - Streamlit Web App
    """)

    st.markdown("### 🧰 Tech Stack")

    st.write("""
- Python  
- Pandas, NumPy  
- NLTK  
- Scikit-learn  
- SHAP (Explainable AI)  
- Streamlit  
    """)

    st.markdown("### 🌍 Real-World Use Cases")

    st.write("""
- Social media moderation  
- News platform verification  
- Fact-checking assistance tools  
- Misinformation detection systems  
    """)

# =========================================================
# 🧾 SIDEBAR
# =========================================================
st.sidebar.title("🧠 System Summary")

st.sidebar.markdown("""
**Model:** Logistic Regression  
**Vectorization:** TF-IDF  
**Accuracy:** ~99%  

💡 Tip: Use full-length articles for best predictions.
""")

# =========================================================
# ⚠️ DISCLAIMER
# =========================================================
st.markdown("""
---
⚠️ **Disclaimer:**  
This system is an AI-based classifier and should not be used as the sole source for verifying news authenticity.
""")