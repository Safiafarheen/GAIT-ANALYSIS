# 🏃 Gait Analysis App

A real-time gait analysis and authentication app built using **Python**, **Streamlit**, **MediaPipe**, **OpenCV**, and **ResNet50**. This project captures and analyzes human gait patterns for biometric authentication and performance feedback — especially for long-distance runners.

---

## 📌 Features

- 🎥 Real-time webcam-based gait data capture
- 🧠 Feature extraction using MediaPipe Pose
- 🔐 Runner authentication using cosine similarity
- 📊 Joint angle variation & efficiency score
- 💡 Personalized performance improvement tips
- 📈 Line charts for knee and ankle Y-coordinates

---

## 🛠 Tech Stack

- **Frontend/UI**: Streamlit
- **Pose Estimation**: MediaPipe
- **Image Processing**: OpenCV
- **Model**: ResNet50 (custom classifier)
- **Database**: JSON (for runner features)
- **Similarity**: Cosine similarity (sklearn)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Safiafarheen/GAIT-ANALYSIS.git
cd GAIT-ANALYSIS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit opencv-python mediapipe torch torchvision matplotlib scikit-learn numpy
```

### 3. Run the App

```bash
streamlit run enhanced_gait_analysis.py
```

---

## 📂 Files

- `enhanced_gait_analysis.py` → Main Streamlit app
- `runner_features.json` → Enrolled runner feature vectors
- `resnet50_custom.pth` → Pretrained model (must be placed in same directory)

---

## ✍️ Author

**Safia Farheen ZR**  
Java Full Stack Developer | AI Enthusiast | Capgemini-TNS Trained  
---

## 📄 License

This project is licensed under the MIT License.
