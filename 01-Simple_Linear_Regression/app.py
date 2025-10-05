import gradio as gr
import joblib

# =========================
# Load pre-trained model
# =========================
model = joblib.load("mpg_model.pkl")

# =========================
# Prediction Function
# =========================
def predict_mpg(weight):
    try:
        weight = float(weight)
        pred = model.predict([[weight]])
        return f"Predicted MPG for {weight:.0f} lbs: {pred[0]:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# Gradio Interface
# =========================
demo = gr.Interface(
    fn=predict_mpg,
    inputs=gr.Number(label="Vehicle Weight (lbs)", precision=0),
    outputs=gr.Textbox(label="Prediction"),
    title="🚗 Auto MPG Predictor",
    description="Enter the weight of a car (in lbs) to predict its fuel efficiency (MPG).",
    live=False
)

# =========================
# Launch App
# =========================
if __name__ == "__main__":
    demo.launch(share=True)
