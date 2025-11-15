# SME IPO Listing Gain Prediction – Machine Learning Project

## 📌 Problem Statement

SME IPOs in India often witness sharp listing-day volatility. Retail investors struggle to assess whether a new SME IPO is likely to provide **listing gains**. This project aims to build a **machine learning model** that helps in decision-making for buying an SME IPO **specifically for listing-day gains and exiting the position**.

---

## 📂 Dataset Description

The dataset includes publicly available SME IPO data scraped from [InvestorGain](https://www.investorgain.com/), containing:

* **Company financials** (Revenue, PAT, Assets)
* **IPO details** (issue size, price band, subscription levels)
* **Market sentiment indicators**
* **Listing price and returns**

The target variable:

* **Listing Gain (%)** – 1 for listing gains else 0

---

## 🔍 Exploratory Data Analysis (EDA) Summary

Key insights from the dataset:

* High correlation between **subscription levels** and listing gains
* Oversubscribed IPOs tend to exhibit stronger gains
* Grey market premium and presence of anchor have highest feature importance
* Clear presence of outliers due to SME market volatility

EDA included:

* Missing value treatment
* Outlier analysis
* Distribution plots
* Correlation heatmap

---

## 🤖 Modeling Approach

Multiple ML models were trained and compared:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

**Evaluation Metrics:**

* macro recall

The best-performing model was selected based on **highest macro recall**.

---

## 🛠️ How to Run the Project Locally

## 🚀 Setup Instructions (Using uv)

To clone the repository and install all dependencies with **uv**, run:

```bash
# Clone the repository
git clone https://github.com/Prajwal-glitch/SME-IPO-listing-gains.git
cd SME-IPO-listing-gains

# Install dependencies using uv
uv sync --locked || uv sync

# Start FastAPI backend
uv run uvicorn predict:app --host 0.0.0.0 --port 9696

# (Optional) Start Streamlit UI
# uv run streamlit run streamlit_app.py

---

## 🐳 Run Using Docker

### **1. Build Docker image**

```bash
docker build -t sme-ipo-ml .
```

### **2. Run container**

```bash
docker run -p 8000:8000 sme-ipo-ml
```

---

## 📡 API Usage Example

### **Endpoint: POST /predict**

```json
{
  "issue_size": 42,
  "subscription": 110.5,
  "revenue_growth": 18.2,
  "pat_margin": 12.5
}
```

#### **Sample Response**

```json
{
  "predicted_listing_gain": 24.7
}
```

---

## ⚠️ Known Limitations & Next Steps

### **Limitations**

* SME market is highly volatile → prediction uncertainty remains high
* Limited dataset availability
* No sentiment or news-based features yet

### **Next Steps**

* Integrate **real-time market sentiment** (Twitter, news, IPO grey market)
* Add **explainable AI** (SHAP values)
* Build Streamlit dashboard
* Deploy full API on Fly.io or Railway

---

## 🏗️ Architecture Diagram

```
   [Raw SME IPO Data]
            |
            v
     [Feature Engineering]
            |
            v
   [Trained ML Model (.pkl)]
            |
            v
      [FastAPI Backend]
            |
            v
        [API /predict]
```

---

## 📘 Author

Prajwal Patil – Machine Learning & Finance Enthusiast

#DataTalksClub
