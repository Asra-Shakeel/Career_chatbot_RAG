import pandas as pd
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV
df = pd.read_csv("2_corpus_chunks.csv")

# Combine searchable text
df["combined"] = df["title"].fillna("") + " " + df["text"].fillna("")

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined"])

def get_answer(question):

    q_vector = vectorizer.transform([question])

    similarity = cosine_similarity(q_vector, X)

    best_score = similarity.max()

    if best_score < 0.10:
        return "Sorry, no relevant answer found."

    best_index = similarity.argmax()

    row = df.iloc[best_index]

    return f"""
Title: {row['title']}

Category: {row['category']}

Risk Level: {row['risk_level']}

Answer:
{row['text']}
"""

demo = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(label="Ask a Career Anxiety Question"),
    outputs="text",
    title="Career Anxiety Chatbot"
)

demo.launch()