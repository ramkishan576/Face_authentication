# 🧠 Face Recognition API with FastAPI

This is a backend API built using **FastAPI** that performs **face recognition** by comparing an uploaded image with pre-existing images stored in folders. The system generates and stores **face embeddings** using the `face_recognition` library and identifies the user if a match is found.

---

## 🚀 Features

- Upload a face image and match it against known profiles.
- Uses **face encodings** (128-dimensional embeddings).
- API Key protected access.
- Automatically precomputes and stores known embeddings.
- Fast performance via in-memory embedding loading.

---

## 📁 Folder Structure

face_recognition_api/
├── known_profiles/
│ ├── user_001/
│ │ ├── image1.jpg
│ │ └── image2.jpg
│ └── user_002/
│ └── image1.jpg
├── main.py
└── known_embeddings.pkl


