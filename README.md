# 🔫 Weapon Detection App 🎯 | YOLOv8 + Streamlit

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green?style=flat-square&logo=github)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-50%25-yellow?style=flat-square)

A real-time weapon detection web application powered by **YOLOv8** and **Streamlit**. Upload an image or video to automatically detect and highlight weapons (guns) using bounding boxes. Perfect for security surveillance, demo projects, or AI learning.

---

## 🖼️ Features

✅ Upload and detect weapons in **images**  
✅ Upload and detect weapons in **videos**  
✅ Real-time **bounding box** visualization  
✅ Displays results with **confidence scores**  
✅ Built with modern deep learning: **YOLOv8**  
✅ Clean and interactive **Streamlit interface**

---

## 🚀 Demo

> Coming Soon – add screenshots or a link to a hosted demo (like Streamlit Cloud or Hugging Face Spaces)

---

## 🧠 Model Details

- **Architecture**: YOLOv8 (Ultralytics)
- **Classes**: `Gun` (extendable)
- **Training Accuracy**: ~50%
- **Framework**: PyTorch + Ultralytics YOLO
- **Model Path**: `model/version3.pt`

---

## 📂 Project Structure

```
Weapon-Detection/
│
├── app.py # Streamlit web application
├── README.md # Project overview
├── LICENSE
├── .gitignore
│
├── model/ # All trained model weights
│ ├── version3.pt # Main YOLOv8 model used
│ ├── gun_detector_final.pt # Other model variants
│ └── yolov8n.pt, yolov8s.pt # Base Ultralytics models
│
├── models/ # Detected crops, test outputs
│ ├── detected_crops/
│ ├── runs/
│ └── vt1_files/
│
├── output/ # Saved image/video results
│ ├── output_video.mp4
│ └── predicted_output1.jpg
│
├── enhance_image/ # (optional image enhancement module)
│
├── notebooks/ # Jupyter notebooks for data prep
│ ├── main.ipynb
│ ├── data_loader.ipynb
│ ├── image_preprocess.ipynb
│ ├── preprocessing.ipynb
│ ├── labels_check.ipynb
│ └── test.ipynb
│
├── src/ # Core training and preprocessing scripts
│ ├── train.py
│ ├── model.py
│ ├── preprocess.py
│ ├── data_loader.py
│ ├── data_generators.py
│ ├── augment.py
│ └── utils.py
│
├── test_data/
│ ├── image/
│ └── video/
│
├── new_env/ # Local environment (optional)
└── requirements.txt # Project dependencies
```

## 🖥️ How to Run the App

### 🔧 Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/weapon-detection-app.git
cd weapon-detection-app
```

### 🐍 Step 2: Set Up Environment
```
python -m venv new_env
.\new_env\Scripts\activate  # On Windows
```

### 📦 Step 3: Install Dependencies
```
pip install -r requirements.txt
```

### 🚀 Step 4: Run Streamlit App
```
streamlit run app.py
```
