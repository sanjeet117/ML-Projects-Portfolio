import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")

def detect(image):
    results = model(image)
    return results[0].plot()

app = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Helmet Detection System",
    description="YOLOv8 Helmet Detection using Custom Trained Model"
)

app.launch()