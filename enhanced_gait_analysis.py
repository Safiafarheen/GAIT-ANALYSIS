
import streamlit as st
import cv2
import torch
import json
import numpy as np
import os
import mediapipe as mp
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "resnet50_custom.pth"
FEATURE_DB = "runner_features.json"
FRAME_LIMIT = 100

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_features(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten(), results.pose_landmarks
    return None, None

def save_runner(name, features):
    if os.path.exists(FEATURE_DB):
        with open(FEATURE_DB, "r") as f:
            db = json.load(f)
    else:
        db = {}
    db[name] = features.tolist()
    with open(FEATURE_DB, "w") as f:
        json.dump(db, f)

def authenticate_runner(uploaded_features):
    with open(FEATURE_DB, "r") as f:
        db = json.load(f)
    best_match, best_score = None, 0
    for name, feats in db.items():
        sim = cosine_similarity([uploaded_features], [feats])[0][0]
        if sim > best_score:
            best_score = sim
            best_match = name
    return best_match if best_score > 0.85 else None

def analyze_performance(features):
    knees_y = features[:, [mp_pose.PoseLandmark.LEFT_KNEE.value * 3 + 1,
                           mp_pose.PoseLandmark.RIGHT_KNEE.value * 3 + 1]]
    ankles_y = features[:, [mp_pose.PoseLandmark.LEFT_ANKLE.value * 3 + 1,
                            mp_pose.PoseLandmark.RIGHT_ANKLE.value * 3 + 1]]
    all_variation = np.std(np.concatenate([knees_y, ankles_y], axis=1))
    efficiency_score = max(0, 100 - all_variation * 100)

    tips = []
    mean_knee = np.mean(knees_y, axis=0)
    mean_ankle = np.mean(ankles_y, axis=0)

    if np.min(mean_knee) < 0.4:
        tips.append("- Raise your knees more for improved stride length.")
    if abs(mean_knee[0] - mean_knee[1]) > 0.1:
        tips.append("- Keep your knee swing symmetrical to reduce imbalance.")
    if np.max(mean_ankle) > 0.6:
        tips.append("- Shorten ankle lift to conserve energy on long runs.")
    return efficiency_score, tips, knees_y, ankles_y

st.title("🎯 Gait Analysis")
mode = st.sidebar.radio("Select Mode", ["Enroll", "Authenticate"])
name = st.text_input("Enter Name") if mode == "Enroll" else None

if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)
    collected, landmarks = [], []
    stframe = st.empty()
    st.info("🎥 Collecting 100 pose frames...")

    while len(collected) < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        features, pose_lm = extract_features(frame)
        if features is not None:
            collected.append(features)
            landmarks.append(pose_lm)
            cv2.putText(frame, f"Pose OK [{len(collected)}/{FRAME_LIMIT}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
    st.success("✅ Camera auto-stopped after collecting 100 valid pose frames.")

    if collected:
        avg_feat = np.mean(collected, axis=0)
        st.info(f"🧠 Total pose frames collected: {len(collected)}")
        if mode == "Enroll" and name:
            save_runner(name, avg_feat)
            st.success(f"✅ Enrolled as '{name}'. Now switch to Authenticate.")
        elif mode == "Authenticate":
            identity = authenticate_runner(avg_feat)
            if identity:
                st.success(f"✅ Recognized as {identity}")
                score, tips, knees_y, ankles_y = analyze_performance(np.array(collected))
                st.markdown(f"### 🏃 Efficiency Score: **{score:.2f}/100**")
                if tips:
                    st.markdown("### 💡 Tips to Improve:")
                    for tip in tips:
                        st.write(tip)
                else:
                    st.info("Excellent gait for endurance running!")

                st.markdown("### 📊 Joint Height Over Time")
                fig, ax = plt.subplots()
                ax.plot(knees_y[:,0], label="Left Knee")
                ax.plot(knees_y[:,1], label="Right Knee")
                ax.plot(ankles_y[:,0], label="Left Ankle")
                ax.plot(ankles_y[:,1], label="Right Ankle")
                ax.set_title("Knee and Ankle Y-Coordinates")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Y Coordinate")
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("❌ Authentication failed.")
    else:
        st.warning("⚠️ No valid pose frames collected.")
