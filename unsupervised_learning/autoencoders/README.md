# Autoencoders (Vanilla)

This project implements a **vanilla autoencoder** using TensorFlow Keras.

## 📌 Description

An autoencoder is a neural network used for **unsupervised learning** that learns a compressed representation (latent space) of input data and reconstructs it back to its original form.

This project builds:
- Encoder network
- Decoder network
- Full autoencoder model

---

## 🧠 Architecture

### Encoder
- Input layer
- Fully connected hidden layers (ReLU)
- Latent space layer

### Decoder
- Mirrors encoder structure (hidden layers reversed)
- Output layer uses **sigmoid activation**

---

## ⚙️ Model Details

- Optimizer: `adam`
- Loss function: `binary_crossentropy`
- Activation (hidden layers): `relu`
- Activation (output layer): `sigmoid`

---

## 📦 Requirements

- Python 3.5
- TensorFlow (Keras API only)
- NumPy
- Ubuntu 16.04 LTS (ALU requirement)

Allowed import:
```python
import tensorflow.keras as keras
📁 Project Structure
unsupervised_learning/autoencoders/
│
├── 0-vanilla.py
├── 0-main.py
└── README.md