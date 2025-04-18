# -*- coding: utf-8 -*-
"""tuned MLP for streamlit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tvM5Aot-3eFcIxjzq-XQ794pCNlZ26i6

# 1. Installing all the libraries
"""

pip install streamlit

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
import streamlit as st

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

"""# 2. Preparing Dataset (test and train)"""

url = "https://drive.google.com/file/d/12SKP4COXVfbn3eOT-9MEH0QC6XwGppf7/view?usp=sharing"
path = "https://drive.google.com/uc?export=download&id="+url.split("/")[-2]
churn = pd.read_csv(path)

churn_df=churn.copy()

#check for duplicates
churn_df.info()

# X and y creation
X = churn_df.drop(columns="CustomerID")
y = X.pop("Churn")

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# need to save test set X_test for later
from google.colab import files

X_test.to_csv("X_test.csv")
files.download("X_test.csv")

"""# 3. Creating Preprocessor for data"""

# select categorical and numerical column names
X_cat_columns = X.select_dtypes(exclude="number").columns
X_num_columns = X.select_dtypes(include="number").columns

# create numerical pipeline
numeric_pipe = StandardScaler()

 # create categorical pipeline, with the OneHotEncoder
categoric_pipe = make_pipeline(
    OneHotEncoder(sparse_output=False,
                  handle_unknown='ignore',
                  drop='if_binary'),
    StandardScaler()
)

categoric_pipe

preprocessor = make_column_transformer(
    (numeric_pipe, X_num_columns),
    (categoric_pipe, X_cat_columns)
)

"""# 4. Full Pipeline with MLP Classifier"""

full_pipeline= make_pipeline(
                preprocessor,
                MLPClassifier(random_state=123,
                              activation='relu',
                              alpha=0.0001,
                              hidden_layer_sizes=(100, 50),
                              learning_rate_init=0.01,
                              solver='adam'))

full_pipeline

full_pipeline.fit(X_train, y_train)

#save the best model in .plk file in the same directory, will be needed for Streamlit
joblib.dump(full_pipeline, 'best_mlp.pkl')

#If you want to run it locally or upload to Streamlit Cloud later
#from google.colab import drive
#drive.mount('/content/drive')

"""# 5. Model Performance

## 5.1. Metrics
"""

# training accuracy

train_pred = full_pipeline.predict(X_train)
accuracy_score(y_train, train_pred)

# testing accuracy

test_pred = full_pipeline.predict(X_test)
accuracy_score(y_test, test_pred)

#confusion matrix from predictions on test data

ConfusionMatrixDisplay.from_predictions(
  y_test,
  test_pred,
  display_labels=['Not Churn', 'Churn']
);

"""Accuracy - how accurate is my prediction.

Precision - when I say it’s Positive , how often am I right

Recall - did I guess all the Positive, or did I miss some.

Cohen’s Kappa coefficient - 1 indicates perfect agreement between the model's predictions and the true classifications, a good Cohen's Kappa is above 0.6
. We have very good coefficient

"""

test_recall = recall_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_cohens_kappa = cohen_kappa_score(y_test, test_pred)

print("\nTest recall:")
print(test_recall)
print("\nTest precision:")
print(test_precision)
print("\nTest f1:")
print(test_f1)
print("\nTest Cohens Kappa:")
print(test_cohens_kappa)

"""## 5.2. The most important features for our Model, that influence accuracy

**Permutation importance** measures how much a model’s performance drops when a feature's values are randomly shuffled. If shuffling a feature significantly decreases the model’s accuracy, it means that the feature was important. If the model's accuracy remains almost the same, the feature was not important.

**How to Interpret the Score?**
High Score (Large Performance Drop) → Feature is important for the model.

Low Score (Small or No Drop) → Feature has little to no importance.

Negative Score → Model performs better when the feature is shuffled (suggesting noise or correlation issues).
"""

# Compute permutation importance
perm_importance = permutation_importance(full_pipeline, X, y, n_repeats=10, random_state=42)

# Create a DataFrame to store feature importances
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Permutation Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (MLP Classifier)")
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance on top
plt.show()

# Display feature importance values
print(importance_df)

"""## 5.3. Using Model to predict probability of churn (with predict_proba)"""

# Extracts only the probability of churn
churn_prob = full_pipeline.predict_proba(X_train)[:, 1]

# Create DataFrame with original index
churn_df = pd.DataFrame(churn_prob, index=X_train.index, columns=["Churn Probability"])

# Format as percentage. ?? why the other is not working?
#churn_df["Churn Probability"] = churn_df["Churn Probability"].apply(lambda x: f"{x:.2%}")
churn_df["Churn Probability"] = churn_df["Churn Probability"].astype(float).round(3)

# Display the first few rows
print(churn_df.head())

"""Checking it, by combining probability and fact"""

#combine initial dataset with churn probability

train_with_churn = X_train.join([churn_df, y_train])
train_with_churn

#checking accuracy
train_with_churn.loc[(train_with_churn['Churn Probability'] < 0.50) & (train_with_churn['Churn']!= 0)]

#other way to combine
# final_df = pd.concat([df1, df2, df3, df4], axis=1)

#now with test data
#Extracts only the probability of churn
churn_prob_test = full_pipeline.predict_proba(X_test)[:, 1]

# Create DataFrame with original index
churn_df_test = pd.DataFrame(churn_prob_test, index=X_test.index, columns=["Churn Probability"])

# Format as percentage
#churn_df["Churn Probability"] = churn_df["Churn Probability"].apply(lambda x: f"{x:.2%}")
churn_df_test["Churn Probability"] = churn_df_test["Churn Probability"].astype(float).round(3)

# Display the first few rows
print(churn_df_test.head())

train_with_churn.loc[(train_with_churn['Churn Probability'] < 0.99) & (train_with_churn['Churn']!= 0)]

"""## 5.4. Confusion Matrix"""

#check with Confusion Matrix
#confusion matrix from predictions

ConfusionMatrixDisplay.from_predictions(
  y_train,
  train_pred,
  display_labels=['Not Churn', 'Churn']
);

#the same number 360 false not churn, so the system counts churn as < 50%

"""# 6. Deployment to Streamlit

First we need to create a tunnel for Streamlit app to make our model accessible locally via the web.
"""

# creating tunnel

import os
import time
from IPython.display import display, HTML
def tunnel_prep():
    for f in ('cloudflared-linux-amd64', 'logs.txt', 'nohup.out'):
        try:
            os.remove(f'/content/{f}')
            print(f"Deleted {f}")
        except FileNotFoundError:
            continue

    !wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -q
    !chmod +x cloudflared-linux-amd64
    !nohup /content/cloudflared-linux-amd64 tunnel --url http://localhost:8501 &
    url = ""
    while not url:
        time.sleep(1)
        result = !grep -o 'https://.*\.trycloudflare.com' nohup.out | head -n 1
        if result:
            url = result[0]
    return display(HTML(f'Your tunnel URL <a href="{url}" target="_blank">{url}</a>'))

"""Next, we create an app.py file that will later be pushed to GitHub and used to deploy the app on Streamlit"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# 
# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# import numpy as np
# 
# from sklearn import set_config
# set_config(transform_output='pandas')
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler
# 
# from sklearn.metrics import accuracy_score
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import make_column_transformer
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.neural_network import MLPClassifier
# from sklearn.inspection import permutation_importance
# from sklearn.datasets import make_classification
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
# 
# # Load your trained model
# model = joblib.load('best_mlp.pkl')
# 
# # Function to make predictions
# def make_predictions(data):
#     predictions = model.predict_proba(data)
#     return predictions
# 
# # Streamlit UI
# st.title("📡 Churn Radar")
# st.subheader("Identifying At-Risk Customers", divider="blue")
# st.write("")  # Adds an empty line
# st.text("Upload your file with customer data (.csv or .xlsx) and identify who needs retention attention – before it’s too late.")
# st.write("")  # Adds an empty line
# # Upload file section
# uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
# if uploaded_file is not None:
#     # Read the dataset
#     if uploaded_file.name.endswith('csv'):
#         data = pd.read_csv(uploaded_file)
#     elif uploaded_file.name.endswith('xlsx'):
#         data = pd.read_excel(uploaded_file)
# 
#     # Show uploaded data
#     st.subheader("Uploaded Data")
#     st.write(data)
# 
#     # Making predictions
#     pred_probs = make_predictions(data) [0:,1]
# 
#     # Add predictions to the original data
#     data['Predicted_Probability'] = pred_probs.round(2)
# 
#     # Bin probabilities
#     bins = [0, 0.25, 0.5, 0.75, 1.0]
#     labels = ['0–0.25', '0.26–0.5', '0.51–0.75', '0.76–1']
#     data['Probability_Bin'] = pd.cut(data['Predicted_Probability'], bins=bins, labels=labels, include_lowest=True)
# 
#     # Show updated dataset with predictions
#     st.write("")  # Adds an empty line
#     st.subheader("Predictions on Your Data")
#     st.text("Scroll to the right ->")
#     st.write(data)
# 
#     # Summary of bins
#     st.write("")  # Adds an empty line
#     st.subheader("Customer Churn Risk Summary")
#     st.write(data['Probability_Bin'].value_counts().sort_index())
#     #here with .sort_index(), we get the output in the logical order of the bins
# 
#     # Visualization
#     #st.subheader("Churn Risk Across the Customer Base")
#     fig, ax = plt.subplots()
# 
#     # Change the figure background color
#     fig.patch.set_facecolor('#ffffff')  # Light grey background for the entire figure
# 
#     # Change the axis background color
#     ax.set_facecolor('#ffffff')  # White background for the plot area
# 
#     # Change the color of the bars using the 'palette' argument
#     sns.countplot(x='Probability_Bin', data=data, order=labels, ax=ax, color=('#048bd4'))  # Using a predefined color palette
# 
#     # Add gridlines to the background
#     ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)  # Customizable gridlines
# 
#     #sns.countplot(x='Probability_Bin', data=data, order=labels, ax=ax, legend='brief',)
#     ax.set_xlabel('Predicted Churn Probability Range')
#     ax.set_ylabel('Number of Customers')
#     st.pyplot(fig)

tunnel_prep()

!streamlit run app.py &>/content/logs.txt &

#from google.colab import files
#files.download('app.py')

# pip freeze
#to see the current versions