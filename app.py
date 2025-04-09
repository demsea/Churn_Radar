
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn import set_config
set_config(transform_output='pandas')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score

# Load your trained model
model = joblib.load('best_mlp.pkl')

# Function to make predictions
def make_predictions(data):
    predictions = model.predict_proba(data)
    return predictions

# Streamlit UI
st.title("📡 Churn Radar")
st.subheader("Identifying At-Risk Customers", divider="blue")
st.write("")  # Adds an empty line
st.text("Upload your file with customer data (.csv or .xlsx) and identify who needs retention attention – before it’s too late.")
st.write("")  # Adds an empty line
# Upload file section
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Read the dataset
    if uploaded_file.name.endswith('csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('xlsx'):
        data = pd.read_excel(uploaded_file)

    # Show uploaded data
    st.subheader("Uploaded Data")
    st.write(data)

    # Making predictions
    pred_probs = make_predictions(data) [0:,1]

    # Add predictions to the original data
    data['Predicted_Probability'] = pred_probs.round(2)

    # Bin probabilities
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['0–0.25', '0.26–0.5', '0.51–0.75', '0.76–1']
    data['Probability_Bin'] = pd.cut(data['Predicted_Probability'], bins=bins, labels=labels, include_lowest=True)

    # Show updated dataset with predictions
    st.write("")  # Adds an empty line
    st.subheader("Predictions on Your Data")
    st.text("Scroll to the right ->")
    st.write(data)

    # Summary of bins
    st.write("")  # Adds an empty line
    st.subheader("Customer Churn Risk Summary")
    st.write(data['Probability_Bin'].value_counts().sort_index())
    #here with .sort_index(), we get the output in the logical order of the bins

    # Visualization
    #st.subheader("Churn Risk Across the Customer Base")
    fig, ax = plt.subplots()

    # Change the figure background color
    fig.patch.set_facecolor('#ffffff')  # Light grey background for the entire figure

    # Change the axis background color
    ax.set_facecolor('#ffffff')  # White background for the plot area

    # Change the color of the bars using the 'palette' argument
    sns.countplot(x='Probability_Bin', data=data, order=labels, ax=ax, color=('#048bd4'))  # Using a predefined color palette

    # Add gridlines to the background
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)  # Customizable gridlines

    #sns.countplot(x='Probability_Bin', data=data, order=labels, ax=ax, legend='brief',)
    ax.set_xlabel('Predicted Churn Probability Range')
    ax.set_ylabel('Number of Customers')
    st.pyplot(fig)
