import numpy as np
import pandas as pd
import re
import string
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], axis=0)
df = df.reset_index(drop=True)

df = df[["title", "text", "label"]]
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df["content"] = df["content"].apply(clean_text)

x = df["content"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.8,
    min_df=2,
    ngram_range=(1,2)
)

xv_train = vectorizer.fit_transform(x_train)

LR = LogisticRegression(max_iter=5000, class_weight="balanced")
LR.fit(xv_train, y_train)

PAC = PassiveAggressiveClassifier(max_iter=1000)
PAC.fit(xv_train, y_train)

SVC = LinearSVC()
SVC.fit(xv_train, y_train)

def predict_news(news):
    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])

    if vectorized.sum() == 0:
        return "⚠ Please enter a longer news article."

    pred_lr = LR.predict(vectorized)[0]
    pred_pac = PAC.predict(vectorized)[0]
    pred_svc = SVC.predict(vectorized)[0]

    final_prediction = round((pred_lr + pred_pac + pred_svc) / 3)

    if final_prediction == 0:
        return "Fake News "
    else:
        return " True News"

interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=10, placeholder="Enter news article here..."),
    outputs="text",
    title="Fake News Detection System",
    description="Enter a news article to check if it is Fake or True."
)

interface.launch()
