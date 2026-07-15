# 💳 Credit Card Fraud Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

# 💳 Credit Card Fraud Detection Using Machine Learning

A machine learning-based credit card fraud detection system that identifies fraudulent transactions using multiple classification algorithms. The project compares the performance of eight machine learning models and selects **XGBoost** as the final model based on evaluation metrics. An interactive **Streamlit web application** was developed for real-time fraud prediction and batch transaction analysis.

---

## 🌐 Live Demo

**Streamlit Application:**  
🔗 https://credit-card-fraud-detection-ml-gtm.streamlit.app/

---

## 📂 GitHub Repository

🔗  https://github.com/Gtm-2003/Credit-Card-Fraud-Detection-ML/

---

## 📌 Project Overview

Credit card fraud is a major challenge in the banking and financial sector. This project aims to detect fraudulent credit card transactions using machine learning techniques. Since the dataset is highly imbalanced, **SMOTE (Synthetic Minority Oversampling Technique)** was used to balance the classes before training.

Eight classification algorithms were trained and evaluated using various performance metrics. Based on the comparison, **XGBoost** was selected as the final model due to its superior overall performance.

A Streamlit web application was built to demonstrate fraud detection through:
- Sample transaction prediction
- Fraud probability estimation
- Adjustable detection threshold
- Batch prediction using CSV upload

---

# 📸 Application Screenshots

## 🏠 Home Page

![Home Page](images/home.png)

---

## ✅ Sample Legitimate Transaction Prediction

![Legitimate Prediction](images/legitimate_prediction.png)

---

## 🚨 Sample Fraudulent Transaction Prediction

![Fraud Prediction](images/fraud_prediction.png)

---

## 📂 Batch Prediction

![Batch Prediction](images/batch_prediction.png)

---

# 🚀 Features

- Credit card fraud detection using Machine Learning
- Data preprocessing and feature scaling
- SMOTE for handling class imbalance
- Comparison of eight classification algorithms
- Hyperparameter tuning using GridSearchCV
- XGBoost final model selection
- Fraud probability prediction
- Adjustable detection threshold
- Confusion Matrix Heatmap
- ROC Curve and ROC-AUC Score
- Classification Report
- Feature Importance Visualization
- Interactive Streamlit Web Application
- Batch transaction prediction using CSV upload
- Downloadable prediction results

---

# 🤖 Machine Learning Algorithms Used

- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Gradient Boosting
- **XGBoost (Final Selected Model)**

---

# 📊 Dataset

**Dataset:** Credit Card Fraud Detection Dataset

### Dataset Information

- Total Transactions: **284,807**
- Legitimate Transactions: **284,315**
- Fraudulent Transactions: **492**

### Features

- Time
- V1 – V28 (PCA-transformed features)
- Amount

### Target Variable

- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**

> **Note:** The deployed Streamlit application uses a lightweight `sample_transactions.csv` file for demonstration purposes, while the machine learning model was trained using the complete dataset.

---

# 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|---------:|----------:|--------:|---------:|
| Logistic Regression | 0.9858 | 0.10 | 0.90 | 0.18 |
| Random Forest | 0.9990 | 0.68 | 0.86 | 0.76 |
| Decision Tree | 0.9991 | 0.75 | 0.74 | 0.75 |
| KNN | 0.9985 | 1.00 | 0.12 | 0.22 |
| Naive Bayes | 0.9926 | 0.14 | 0.66 | 0.23 |
| SVM | 0.9983 | 0.00 | 0.00 | 0.00 |
| Gradient Boosting | 0.9983 | 0.53 | 0.18 | 0.27 |
| **XGBoost** | **0.9994** | **0.88** | **0.78** | **0.83** |

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn
- Imbalanced-learn (SMOTE)

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Gtm-2003/Credit-Card-Fraud-Detection-ML.git
```

Move into the project directory:

```bash
cd Credit-Card-Fraud-Detection-ML
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
Credit-Card-Fraud-Detection-ML/
│
├── app.py
├── fraud_detection.ipynb
├── xgb_model.pkl
├── sample_transactions.csv
├── requirements.txt
├── README.md
└── images/
    ├── home.png
    ├── legitimate_prediction.png
    ├── fraud_prediction.png
    └── batch_prediction.png
```

---

# 🎯 Project Outcomes

- Successfully developed an end-to-end credit card fraud detection system.
- Compared eight machine learning algorithms.
- Selected XGBoost as the best-performing model.
- Built and deployed an interactive Streamlit web application.
- Implemented batch prediction for multiple transactions.
- Achieved excellent fraud detection performance using ROC-AUC and evaluation metrics.

---

# 🔮 Future Enhancements

- Real-time fraud detection using transaction streams.
- Integration with banking systems through REST APIs.
- Deep learning-based fraud detection models.
- Cloud deployment with automated monitoring.
- Explainable AI (XAI) techniques for prediction interpretation.

---

# 👨‍💻 Author

**Gowtham K R**

BCA Student | Python Developer | Machine Learning Enthusiast

---

⭐ If you found this project helpful, consider giving the repository a **Star** on GitHub.
