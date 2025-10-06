# app.py
import gradio as gr
import joblib

# =========================
# Load the trained model
# =========================
# Make sure 'mpg_model.pkl' is in the same folder as this file
model = joblib.load("mpg_mlr_model.pkl")

# =========================
# Prediction Function
# =========================
def predict_mpg(cylinders, displacement, weight):
    try:
        # Model expects a 2D array: [[cylinders, displacement, weight]]
        prediction = model.predict([[cylinders, displacement, weight]])
        return f"🔮 Predicted MPG: {prediction[0]:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# Gradio Interface
# =========================
iface = gr.Interface(
    fn=predict_mpg,
    inputs=[
        gr.Number(label="Cylinders"),
        gr.Number(label="Displacement"),
        gr.Number(label="Weight (lbs)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🚗 MPG Predictor",
    description="Enter car specifications to predict Miles Per Gallon using a trained Linear Regression model."
)

# =========================
# Launch the App
# =========================
if __name__ == "__main__":
    iface.launch(share=True, inline=False)
