import streamlit as st
import pandas as pd
import numpy as np
# import joblib
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data Collection
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff'
data = requests.get(data_url).text
data = data.split('\n')
data = [i for i in data if not i.startswith('%')]
df = pd.DataFrame([i.split(',') for i in data])
data = [i for i in data if not i.startswith('%') and not i.startswith('@')]

df = df.iloc[:, :-1]
df.columns = ["having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol", "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length", "Favicon", "port",
              "HTTPS_token", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
              "Abnormal_URL", "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord",
              "web_traffic",
              "Page_Rank", "Google_Index", "Links_pointing_to_page", "Statistical_report"]
df['Result'] = df['Statistical_report']
df = df.replace({'Result': -1}, 0)

# Convert all columns to numeric data types
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Convert all columns to integer data types
df = df.astype(int)

# Data Preprocessing
X = df.drop(['Result'], axis=1)
y = df['Result']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Model Testing
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

st.title("Phishing Detection System")

st.title("Phishing Detection System")


# Function to extract features from a website URL
def extract_features(url):
    try:
        domain = urlparse(url).netloc
        http = 1 if urlparse(url).scheme == 'https' else 0
        soup = BeautifulSoup(requests.get(url).content, 'html.parser')
        anchor_count = len(soup.find_all('a'))
        forms_count = len(soup.find_all('form'))
        iframes_count = len(soup.find_all('iframe'))
        popups_count = len(soup.find_all('popup'))
        symbols_count = len(re.findall(r'[!$*,:;]', url))
        features = [http, len(url), anchor_count, forms_count, iframes_count, popups_count, symbols_count]
        return features
    except:
        return None


# User Interface
url = st.text_input("Enter a website URL:")

if url:
    features = extract_features(url)
    if features:
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]

        if prediction == 0:
            st.error("This is a phishing website!")
        else:
            st.success("This is a legitimate website.")
