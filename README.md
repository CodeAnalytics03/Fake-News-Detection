# 🧠 Fake News Intelligence System

🚀 **Live App:** https://fake-news-detection-tharunesh-ds-1st-project.streamlit.app

---

## 📌 Overview

The **Fake News Intelligence System** is an end-to-end Natural Language Processing (NLP) and Machine Learning application designed to classify news articles as **Real** or **Fake**.

This project demonstrates a **production-style data science workflow**, integrating data preprocessing, feature engineering, model building, and cloud deployment into a unified system.

---

## 🎯 Key Features

* 🔍 Real-time fake news prediction
* 🧠 NLP-based text processing (TF-IDF)
* ⚙️ Machine Learning model (Logistic Regression)
* 🌐 Interactive web interface using Streamlit
* 🎨 Dark/Light mode UI
* ⚡ Fast and lightweight inference

---

## 🏗️ System Architecture

```text
User Input (Text)
        ↓
Text Preprocessing (Cleaning, Tokenization)
        ↓
Feature Extraction (TF-IDF Vectorization)
        ↓
Machine Learning Model (Logistic Regression)
        ↓
Prediction Output (Real / Fake)
        ↓
Streamlit UI Display
```

---

## 🛠️ Tech Stack

### 🔹 Programming & Libraries

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK

### 🔹 Machine Learning

* Logistic Regression
* TF-IDF Vectorizer

### 🔹 Visualization

* Matplotlib
* Seaborn

### 🔹 Deployment

* Streamlit
* GitHub

---

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── app/
│   ├── app.py                # Streamlit application
│   ├── styles.css           # UI styling
│   └── models/              # Model artifacts (required for deployment)
│
├── notebooks/               # Development & experimentation
│   ├── 01_data_loading.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_building.ipynb
│
├── src/                     # Core logic (optional modules)
│
├── requirements.txt         # Dependencies
├── runtime.txt              # Python version
├── README.md
└── .gitignore
```

---

## ⚙️ How It Works

1. User enters a news article
2. Text is preprocessed:

   * Lowercasing
   * Stopword removal
   * Tokenization
3. TF-IDF converts text into numerical features
4. Logistic Regression model predicts:

   * ✅ Real News
   * 🚫 Fake News
5. Result is displayed instantly in the UI

---

## 📊 Model Details

| Component      | Value                 |
| -------------- | --------------------- |
| Algorithm      | Logistic Regression   |
| Feature Method | TF-IDF                |
| Problem Type   | Binary Classification |
| Accuracy       | ~99%                  |

---

## 🚀 Getting Started (Local Setup)

### 1. Clone Repository

```bash
git clone https://github.com/CodeAnalytics03/fake-news-detection.git
cd fake-news-detection
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the App

```bash
streamlit run app/app.py
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

🔗 **Live Demo:**
https://fake-news-detection-tharunesh-ds-1st-project.streamlit.app

---

## ⚠️ Important Notes

* Model files must be present in `app/models/`
* Large datasets are excluded from the repository
* Ensure correct file paths when deploying

---

## 📈 Future Enhancements

* 🔮 Transformer-based models (BERT, RoBERTa)
* 📊 Confidence score visualization
* 🧠 Explainability (SHAP/LIME)
* 🌍 Multilingual fake news detection
* 📱 Mobile-responsive UI

---

## 🙌 Acknowledgements

* Open-source NLP and ML libraries
* Streamlit for rapid deployment
* Dataset sources for fake news classification

---

## 👨‍💻 Author

**Tharunesh R**
Aspiring Data Scientist | NLP Enthusiast

---

## ⭐ If you found this useful

Give this repo a ⭐ and share it!
