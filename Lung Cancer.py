#streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the page title and header
st.set_page_config(page_title='Lung cancer analysis', page_icon='::')
st.title('LUNG Cancer and prediction')


# Display the cover image
cover_image = "image.jpg"
st.image(, use_column_width=True)


        
# Set the title and link of the sidebar
st.sidebar.title("Helma Falahati ")
st.sidebar.markdown("[GitHub](https://github.com/helmaft)")


# Load the data
df = pd.read_csv('cancer patient data sets.csv')



# Main Function
def main():
    st.title('Lung Cancer Analysis and Prediction')
    
    df = load_data()
    
    # Display the raw data
    st.subheader('Raw Data')
    st.write(df.head())

    # Data Preprocessing and EDA
    if st.checkbox('Show EDA'):
        st.subheader('Exploratory Data Analysis')
        st.write(df.describe())
        st.write(df.info())
        st.write(df.groupby('Gender').describe())

        # Visualizations
        st.subheader('Visualizations')
        fig, ax = plt.subplots()
        sns.countplot(x='Gender', data=df, ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=ax)
        st.pyplot(fig)

    # Data Preprocessing for Model
    X = df.drop(columns=['Level', 'Patient Id'])
    y = df['Level']

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = label_encoder.fit_transform(X[col])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    if st.checkbox('Train Model'):
        st.subheader('Model Training and Evaluation')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy:', accuracy)

        st.write('Confusion Matrix:')
        st.write(confusion_matrix(y_test, y_pred))

        st.write('Classification Report:')
        st.write(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
