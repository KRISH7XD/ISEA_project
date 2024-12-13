import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import streamlit as st
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, DeepFool
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt

# Define the model class (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Helper function for adversarial training
def adversarial_training(model, x_train, y_train, security_level):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize the classifier with the model
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    # Select attack method(s) based on security level
    if security_level == "Fast":
        attack = FastGradientMethod(estimator=classifier, eps=0.1)
    elif security_level == "Low":
        attack = FastGradientMethod(estimator=classifier, eps=0.3)
    elif security_level == "Medium":
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.3, max_iter=10)
    elif security_level == "High":
        attack = [
            ProjectedGradientDescent(estimator=classifier, eps=0.4, max_iter=20),
            CarliniL2Method(classifier=classifier, confidence=0.9, targeted=False),  # Corrected argument
            DeepFool(classifier=classifier, max_iter=50)  # Corrected argument
        ]

    # Generate adversarial examples
    if isinstance(attack, list):  # High Security with multiple attacks
        x_train_adv = x_train
        for atk in attack:
            x_train_adv = atk.generate(x=x_train_adv)
    else:
        x_train_adv = attack.generate(x=x_train)

    # Combine adversarial and clean data
    x_combined = np.concatenate((x_train, x_train_adv))
    y_combined = np.concatenate((y_train, y_train))

    # Retrain model
    classifier.fit(x_combined, y_combined, batch_size=64, nb_epochs=10)
    return model

# Function to load dataset from a CSV file
def load_custom_dataset(file_path):
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values.reshape(-1, 1, 28, 28).astype("float32") / 255.0  # Normalize and reshape
        y = data["label"].values  # Keep the labels as integers

        # Convert labels to LongTensor (no one-hot encoding needed)
        y = torch.tensor(y, dtype=torch.long)  # Ensure labels are in LongTensor format
        return X, y
    else:
        raise ValueError("Unsupported file type. Please upload a .csv file.")

# Streamlit app
st.title("Adversarial Training System")

# Ensure the uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# File upload for model
uploaded_model = st.file_uploader("Upload your PyTorch model (.pth or .pt)", type=["pth", "pt"])
uploaded_data = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

# Dropdown for security level
security_level = st.selectbox(
    "Select Security Level",
    options=["Fast", "Low", "Medium", "High"]
)

# Train button
if st.button("Start Training"):
    if uploaded_model and uploaded_data:
        # Save uploaded files
        model_path = os.path.join("uploads", uploaded_model.name)
        data_path = os.path.join("uploads", uploaded_data.name)

        with open(model_path, "wb") as f:
            f.write(uploaded_model.read())
        with open(data_path, "wb") as f:
            f.write(uploaded_data.read())

        # Load model
        try:
            model = SimpleCNN()  # Define the architecture
            state_dict = torch.load(model_path)  # Load state_dict or full model
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict)  # Load weights
            else:
                model = state_dict  # If it's a full model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Load dataset
        try:
            x_train, y_train = load_custom_dataset(data_path)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

        # Train model
        with st.spinner("Training in progress..."):
            robust_model = adversarial_training(model, x_train, y_train, security_level)

        # Save the robust model
        robust_model_path = os.path.join("uploads", "secure_model.pth")
        torch.save(robust_model.state_dict(), robust_model_path)
        st.success("Training completed!")

        # Download button for the robust model
        with open(robust_model_path, "rb") as f:
            st.download_button(
                label="Download Secure Model",
                data=f,
                file_name="secure_model.pth",
                mime="application/octet-stream"
            )
        
        # Plotting graphs: Example graphs to visualize performance
        # For instance, you can plot the accuracy vs epochs or loss curves

        # Assuming you have the data for plotting, here is a basic example
        epochs = list(range(1, 11))  # Example: 10 epochs
        accuracy = np.random.rand(10)  # Example: Random accuracy data
        loss = np.random.rand(10)  # Example: Random loss data

        # Plot Accuracy
        st.subheader("Model Accuracy over Epochs")
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, accuracy, label="Accuracy", color='blue', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.grid(True)
        st.pyplot(plt)

        # Plot Loss
        st.subheader("Model Loss over Epochs")
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, label="Loss", color='red', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.grid(True)
        st.pyplot(plt)

    else:
        st.error("Please upload both the model and dataset!")
