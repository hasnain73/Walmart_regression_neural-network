"""
Walmart Weekly Sales Predictor - Gradio App
Hugging Face Spaces Deployment
"""

import gradio as gr
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# Load model and preprocessors
print("Loading model and preprocessors...")
model = keras.models.load_model('models/model.h5', compile=False)

scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Try to load label encoders if they exist
try:
    label_encoders = joblib.load('models/label_encoders.pkl')
except:
    label_encoders = None

print(f"Model loaded successfully!")
print(f"Features expected: {feature_names}")
print(f"Number of features: {len(feature_names)}")

def predict_sales(*inputs):
    """
    Predict weekly sales based on input features
    """
    try:
        # Create input array
        input_data = np.array([list(inputs)])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)[0][0]
        
        # Format output
        result = f"💰 **Predicted Weekly Sales: ${prediction:,.2f}**"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
print("Creating Gradio interface...")

# Create inputs dynamically based on features
inputs = []
for feature in feature_names:
    inputs.append(
        gr.Number(label=feature, value=0)
    )

# Create interface
demo = gr.Interface(
    fn=predict_sales,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result", lines=2),
    title="🏪 Walmart Weekly Sales Predictor",
    description="""
    ### Simple Neural Network Regression Model
    
    This model predicts weekly sales for Walmart stores based on various features.
    
    **How to use:**
    1. Enter values for all features
    2. Click Submit
    3. View predicted weekly sales
    
    **Model Details:**
    - Architecture: Simple Neural Network (64→32→1 neurons)
    - Training: 20 epochs, MSE loss
    - Framework: TensorFlow/Keras
    """,
    examples=[
        # Add some example inputs (adjust based on your features)
        # You can add real examples after seeing the actual data
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Gradio App...")
    print("="*60)
    demo.launch()
