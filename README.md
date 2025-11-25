# AI-Based Credit Card Fraud Detection System

A Streamlit web application that predicts fraudulent credit card transactions using machine learning.  
The app includes data preprocessing, oversampling, model training, evaluation, and real-time prediction.  
Works with the well-known **creditcard.csv** dataset from Kaggle.

---

## 🚀 Features

- Load and explore the credit card fraud dataset  
- Interactive EDA with Plotly visualizations  
- Oversampling to fix class imbalance  
- Train two machine learning models:
  - Logistic Regression  
  - Random Forest  
- Metrics shown:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - ROC-AUC  
- Confusion matrix heatmaps  
- Predict fraud for a single transaction  
- Clean UI built with Streamlit  

---

## 📂 Project Structure

.
├── app.py
├── creditcard.csv # Add manually
├── requirements.txt
└── README.md



---

## 🛠️ Tech Stack

- Python 3  
- Streamlit  
- Scikit-Learn  
- Pandas  
- NumPy  
- Plotly  
- StandardScaler  
- RandomForestClassifier  
- LogisticRegression  

---

## Dataset

This project uses the publicly available **Credit Card Fraud Detection Dataset** from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv

Download the dataset from Kaggle and place the `creditcard.csv` file inside the project directory before running the app.

---

## 🔧 Installation

### 1. Clone the repository

git clone https://github.com/RiyanshiVerma-11/Credit-Card-Fraud-detection-System.git
cd Credit-Card-Fraud-detection-System


### 2. Create a virtual environment

python -m venv venv


### 3. Activate the environment

**Windows:**
venv\Scripts\activate


**Mac/Linux:**
source venv/bin/activate

### 4. Install dependencies

pip install -r requirements.txt


### 5. Add dataset

Place **creditcard.csv** in the root folder.

---

## ▶️ Run the Application

streamlit run app.py


---

## 📈 How It Works

1. Load dataset  
2. Run interactive EDA  
3. Oversample fraud cases  
4. Train selected machine learning models  
5. Evaluate multiple performance metrics  
6. Predict fraud for a manually entered transaction  

---

## ⭐ Future Enhancements

- Add more ML models like XGBoost, SVM, and Neural Networks  
- Auto hyperparameter tuning  
- Deploy the project online  
- Add additional visual analysis  

---

## 📝 License

This project is licensed under the **MIT License**.

---

# requirements.txt

streamlit
pandas
numpy
scikit-learn
plotly

