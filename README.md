# 📉 Churn_Radar – Predicting Customer Churn with Machine Learning

**Churn_Radar** is a Streamlit-powered machine learning web app that predicts the probability of customer churn based on user behavior and demographics such as preferred login device, order preferences, marital status, tenure, and more.

---

## 🧠 Why This Project Matters

For businesses, **customer churn** is one of the biggest challenges — losing customers can directly impact profitability. My final project focuses on predicting churn using machine learning, aiming to help companies retain valuable customers more effectively.

By analyzing a real-world dataset from Kaggle, I developed a predictive model that identifies customers at risk of leaving. With this insight, businesses can proactively take action to improve customer retention, personalize outreach, and make data-driven decisions.

---

## 📁 Project Structure

This repository includes all steps from data cleaning to model deployment:

| File/Folder                   | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `1_cleaning_dataset.ipynb`   | Cleans the raw dataset, imputes missing values using medians, and performs feature engineering. |
| `2_models_comparison.ipynb`  | Compares different supervised ML models (e.g., Logistic Regression, Random Forest, MLP) using accuracy, precision, F1-score, and Cohen's Kappa. Performs hyperparameter tuning with GridSearchCV. |
| `3_tuned_mlp_for_streamlit.ipynb` | Trains the best-performing model — an MLP Classifier with 98% test accuracy — and saves it for deployment. |
| `best_mlp.pkl`               | Pre-trained MLPClassifier model. Ready to use without retraining, saving time and computation. |
| `app.py`                     | The main file for deploying on Streamlit app that loads the trained model and offers an interactive UI for churn prediction. |
| `e_comm_churn.csv`           | The original dataset downloaded from Kaggle.                                                   |
| `churn_cleaned.csv`          | Dataset after cleaning and preprocessing.                                                      |
| `X-test.csv`                 | A test dataset used to simulate predictions via the Streamlit app.                             |

---

## ⚙️ How It Works

1. Upload your customer data file (`.csv` or `.xlsx`) via the app.
2. The app uses the pre-trained model to calculate each customer’s probability of churning.
3. It displays the updated Dataset with churn probability for each customer and a visualization to help identify customers who need retention attention (updated dataset can be downloaded in .csv format).

---

## 📊 Live Demo

You can try out the model here:  
🔗 **[ChurnRadar on Streamlit](https://churnradar.streamlit.app/)**

---

## 📦 Tech Stack

- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- Streamlit
- Jupyter Notebook
- Joblib
- Supervised Machine Learning Models and data preprocessing techniques

---

## 📬 Get in Touch

Feel free to reach out if you have suggestions or want to collaborate.  
Happy analyzing! 😊
