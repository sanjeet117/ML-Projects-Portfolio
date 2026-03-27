import gradio as gr
import pickle
import re
import nltk
from nltk.corpus import stopwords

# setup
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# SAME preprocessing as training
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

# prediction
def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    return f"Sentiment: {result}"

# UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter review here..."),
    outputs="text",
    title="Sentiment Analysis App",
    description="Predict Positive or Negative sentiment"
)

interface.launch()