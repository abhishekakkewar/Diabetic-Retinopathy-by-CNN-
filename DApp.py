#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18  # Import your model or use a pre-trained model

# Load the model
def load_model(model_path):
    model = torch.load("Retino_model.pt")
    model.eval()
    return model

# Define image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define prediction function
def predict(image, model):
    image = transform_image(image)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities

# Streamlit UI
def main():
    st.title("Diabetic Retinopathy Diagnosis")
    
    st.write("Upload an image of a retina scan to predict if the patient has diabetic retinopathy.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load model
        model = load_model("Retino_model.pt")
        
        # Predict
        predicted_class, probabilities = predict(image, model)
        
        # Show result
        class_labels = ['Diabetic Retinopathy', 'No Diabetic Retinopathy']
        st.write(f"Prediction: {class_labels[predicted_class]}")
        st.write(f"Probability: {probabilities[0][predicted_class].item() * 100:.2f}%")

if __name__ == "__main__":
    main()


# In[ ]:




