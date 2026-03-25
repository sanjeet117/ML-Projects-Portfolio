import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("cat_dog_model.keras")

def predict(img):
    img = img.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        return f"Dog ({confidence:.2f})"
    else:
        return f"Cat ({1-confidence:.2f})"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cat vs Dog Classifier "
)

interface.launch()