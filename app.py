Group members :- Kathrotiya Jenis , naanaa Hedi , Ryan Ghaoui


import json
from pathlib import Path
import joblib
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
labels = metadata["topics"]

tfidf_model = joblib.load(MODEL_DIR / "tfidf_model.joblib")
embedding_classifier = joblib.load(MODEL_DIR / "embedding_classifier.joblib")
encoder = SentenceTransformer(metadata["embedding_model"])

def classify(article, model_choice):
    if not article or len(article.strip()) < 20:
        return {"Please enter a longer article (at least 20 characters).": 1.0}

    if model_choice == "TF-IDF + Logistic Regression":
        probs = tfidf_model.predict_proba([article])[0]
    else:
        vector = encoder.encode([article], normalize_embeddings=True)
        probs = embedding_classifier.predict_proba(vector)[0]

    return {labels[i]: float(probs[i]) for i in np.argsort(probs)[::-1]}

with gr.Blocks(title="African News Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# African News Classifier")
    gr.Markdown(
        "Paste a news article to predict its topic. The models were trained "
        "on MasakhaNEWS data from 16 language configurations."
    )
    gr.Markdown("**Topics:** " + ", ".join(labels))
    article = gr.Textbox(
        label="News article",
        lines=10,
        placeholder="Paste the headline and article text..."
    )
    model_choice = gr.Radio(
        ["TF-IDF + Logistic Regression",
         "Multilingual embeddings + Logistic Regression"],
        value="Multilingual embeddings + Logistic Regression",
        label="Choose model"
    )
    button = gr.Button("Classify article", variant="primary")
    output = gr.Label(label="Predicted topic probabilities", num_top_classes=7)
    button.click(classify, inputs=[article, model_choice], outputs=output)
    gr.Markdown(
        "**Limitations:** Model scores are estimates, not verified facts. "
        "Performance may vary by language, topic and source."
    )

if __name__ == "__main__":
    demo.launch(share=True)
