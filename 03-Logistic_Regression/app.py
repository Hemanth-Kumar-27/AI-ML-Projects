import gradio as gr
import pandas as pd
import pickle

# Load the trained model from pickle file
with open("breast_cancer_model.pkl", "rb") as f:
    logistic_fit = pickle.load(f)

# Define feature columns (same as training)
cols = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

# ✅ Prediction function
def predict_breast_cancer(*values):
    df = pd.DataFrame([values], columns=cols)
    prob = logistic_fit.predict(df)[0]
    label = "Malignant" if prob > 0.5 else "Benign"
    return f"🩺 Prediction: {label}\n\nProbability of Malignant: {prob:.3f}"

# ✅ Gradio Interface
inputs = [gr.Number(label=col) for col in cols]

iface = gr.Interface(
    fn=predict_breast_cancer,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result"),
    title="Breast Cancer Prediction (Logistic Regression)",
    description="Enter the 30 feature values to predict whether the tumor is benign or malignant.",
    theme="soft",
    allow_flagging="never"
)

iface.launch(debug=True)