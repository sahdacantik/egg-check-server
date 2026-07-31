# 🖥 Egg Check Server

Backend API for the Egg Check Mobile application, providing AI-powered egg quality inspection using Computer Vision and Convolutional Neural Networks (CNN).

This server is responsible for image preprocessing, CNN inference, Grad-CAM generation, and REST API communication with the Flutter mobile application.

> This project was developed as part of my Bachelor's Thesis in Computer Science at Universitas Pakuan.

---

## ✨ Features

- 🧠 CNN-based egg crack classification
- 🥚 Single-egg and multi-egg inspection
- ⚙️ Automatic preprocessing pipeline
- 🔥 Grad-CAM visualization
- 🌐 REST API using Flask
- ☁️ Cloud deployment with Railway
- 📤 JSON response for Flutter application

---

## 🏗 System Architecture

```
Flutter App
      │
      │ HTTP Request
      ▼
Flask REST API
      │
Automatic Preprocessing Pipeline
      │
CNN Model (MobileNetV2)
      │
Grad-CAM Generation
      │
JSON Response
      ▼
Flutter Application
```

---

## ⚙ Tech Stack

### Backend

- Python
- Flask

### AI

- TensorFlow
- MobileNetV2
- OpenCV
- NumPy

### Deployment

- Railway

### Other

- REST API
- JSON
- Base64 Encoding

---

## 🚀 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict single or multiple eggs |
| `/predict_single` | POST | Detail inspection from one viewing angle |

---

## 🧠 Automatic Preprocessing Pipeline

Before inference, every uploaded image is automatically processed through several preprocessing stages:

- Background segmentation
- CLAHE enhancement
- Noise reduction
- Egg localization
- HSV tray detection
- Hough Circle detection
- Image cropping
- Image resizing
- Normalization

This pipeline ensures that every image is processed consistently before being passed to the CNN model.

---

## 🥚 Multi-Egg Inspection

The backend automatically detects egg trays using HSV masking and identifies individual eggs using Hough Circle Transform.

Each detected egg is:

1. Cropped
2. Preprocessed
3. Classified individually
4. Returned as structured JSON

Supports up to **30 eggs** in one image. :contentReference[oaicite:0]{index=0}

---

## 🔍 Detail Check

For uncertain predictions, the backend provides a lightweight endpoint dedicated to multi-angle inspection.

The mobile application sends up to three images from different viewing angles.

The backend:

- preprocesses every image
- predicts crack probability
- generates Grad-CAM
- returns the highest crack probability as the final decision

This approach improves reliability while reducing unnecessary image capture through an early-exit mechanism. :contentReference[oaicite:1]{index=1}

---

## 🔥 Grad-CAM Visualization

To improve model interpretability, the server generates Grad-CAM heatmaps for every prediction.

The generated visualization highlights image regions contributing most to the CNN prediction, helping users better understand the model's decision-making process. :contentReference[oaicite:2]{index=2}

---

## 📱 Mobile Application

This backend serves the Flutter application available in:

➡️ **[Egg Check Mobile](https://github.com/sahdacantik/egg_check_mobile)**

---

## 👨‍💻 Author

Sahda Rahani Susilawati

• Computer Science • Software Engineer • Computer Vision
