import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")

def predict(img):
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = model.predict(img)[0][0]

    return "Mask" if pred < 0.5 else "No Mask"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Face Mask Detection",
    description="Upload image or use webcam"
)

interface.launch()