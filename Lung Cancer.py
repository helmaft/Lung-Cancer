import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px

# Page configuration
st.set_page_config(page_title='Lung Cancer Analysis', page_icon='🫁', layout='wide')

# Sidebar
with st.sidebar:
    st.title("Lung Cancer Analysis")
    st.image("image.jpg", use_column_width=True)
    st.markdown("### Helma Falahati")
    st.markdown("[GitHub](https://github.com/helmaft)")
    st.info('This app performs lung cancer analysis and prediction using patient data.')

# Main page
st.title('Lung Cancer Analysis and Prediction')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cancer patient data sets.csv')
    return df

df = load_data()

# Data Overview
st.header('Data Overview')
st.write(df.head())
st.write(f"Dataset shape: {df.shape}")

# Data Preprocessing
@st.cache_data
def preprocess_data(df):
    X = df.drop(columns=['Level', 'Patient Id'])
    y = df['Level']
    
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = label_encoder.fit_transform(X[col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler

X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)

# EDA
st.header('Exploratory Data Analysis')
eda_option = st.selectbox('Choose EDA option', ['Summary Statistics', 'Data Distribution', 'Correlation Heatmap'])

if eda_option == 'Summary Statistics':
    st.write(df.describe())
elif eda_option == 'Data Distribution':
    feature = st.selectbox('Select feature for distribution', X.columns)
    fig = px.histogram(df, x=feature, color='Level', marginal='box')
    st.plotly_chart(fig)
elif eda_option == 'Correlation Heatmap':
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, fmt='.2f', ax=ax)
    st.pyplot(fig)

# Modeling
st.header('Model Training and Evaluation')
model_option = st.selectbox('Choose a model', ['Logistic Regression', 'Random Forest', 'SVM'])

@st.cache_resource
def train_model(model_name, X_train, y_train):
    if model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    else:
        model = SVC(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(model_option, X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
plt.ylabel('Actual')
plt.xlabel('Predicted')
st.pyplot(fig)

st.subheader('Classification Report')
st.text(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
st.write(f'Cross-validation scores: {cv_scores}')
st.write(f'Mean CV score: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})')

# Feature Importance (for Random Forest)
if model_option == 'Random Forest':
    st.subheader('Feature Importance')
    importances = pd.DataFrame({'feature':X.columns, 'importance':model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False)
    fig = px.bar(importances, x='feature', y='importance')
    st.plotly_chart(fig)

# Prediction Interface
st.header('Predict Lung Cancer Level')
user_input = {}
for column in X.columns:
    user_input[column] = st.number_input(f"Enter {column}", value=0.0)

input_df = pd.DataFrame([user_input])

if st.button('Predict'):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.write(f'The predicted lung cancer level is: {prediction[0]}')

# About
st.sidebar.header('Info')
st.sidebar.info('This Streamlit app is designed to analyze lung cancer data and predict cancer levels. '
                'It uses machine learning models to make predictions based on patient data. ')
