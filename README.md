🧠 Brain Tumor Detection using MERN + Machine Learning

A full-stack MERN + ML web application that detects and classifies brain tumors from MRI images using a Convolutional Neural Network (CNN).

⚠️ This project is for educational and research purposes only. It is not a medical diagnostic tool.

🚀 Tech Stack
🧠 Machine Learning

Python

TensorFlow / Keras

OpenCV

NumPy

CNN (Custom Architecture)

🌐 Backend

Node.js

Express.js

REST API

💻 Frontend

React.js

🗄 Database

MongoDB

🏗 Project Architecture
Frontend (React)
        ↓
Backend (Node + Express)
        ↓
ML Service (Flask API - TensorFlow Model)
        ↓
Prediction Response (Tumor Type + Confidence)


The ML model runs as a separate microservice and communicates with the Node backend via REST API.

🧠 Problem Statement

This project aims to classify MRI brain images into one of the following categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The model is trained on labeled MRI image data and predicts tumor type along with confidence score.

📂 Project Structure
BrainTumer/
│
├── backend/              # Node + Express backend
├── frontend/             # React frontend
├── ml_service/           # Python ML microservice
│   ├── dataset/          # (Not pushed to GitHub)
│   ├── saved_model/      # (Not pushed to GitHub)
│   ├── app.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── requirements.txt
│
├── .gitignore
└── README.md

📊 Dataset

This project uses a public Brain Tumor MRI dataset from Kaggle.

🔗 Dataset Link:
(Replace with actual Kaggle link)

⚠️ The dataset is not included in this repository.

After downloading, place it inside:

ml_service/dataset/


Folder structure should be:

dataset/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/

🧪 How to Run the ML Service
1️⃣ Navigate to ML service
cd ml_service

2️⃣ Create virtual environment
py -m venv venv


Activate (Windows PowerShell):

.\venv\Scripts\Activate.ps1

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model
py train.py


Model will be saved inside:

ml_service/saved_model/model.h5

5️⃣ Start ML API
py app.py

🧠 Model Architecture

Custom CNN Architecture:

Conv2D → ReLU

MaxPooling

Conv2D → ReLU

MaxPooling

Conv2D → ReLU

MaxPooling

Flatten

Dense (128)

Dropout

Softmax (4 classes)

The model is trained using:

Categorical Crossentropy

Adam Optimizer

Data Augmentation

🔌 Backend Setup
cd backend
npm install
npm start

💻 Frontend Setup
cd frontend
npm install
npm start

📈 Future Improvements

Grad-CAM for explainable AI

Tumor segmentation (U-Net)

Docker containerization

Cloud deployment (Render / AWS)

Model performance dashboard

👨‍💻 Author

Harshit
GitHub: https://github.com/harshitWhoCde