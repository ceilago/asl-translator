# streamlit_app.py

import os
import streamlit as st
import tempfile
import cv2
import json
import mediapipe as mp
import openai
from typing import List

# --- Configuration ---
FRAMES_PER_SECOND = 15
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set

# --- Frame Extraction ---
def extract_frames(video_path: str, fps: int) -> List[str]:
    output_paths = []
    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)
    count = 0
    saved = 0

    tmp_dir = tempfile.mkdtemp()
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % interval == 0:
            frame_path = os.path.join(tmp_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, image)
            output_paths.append(frame_path)
            saved += 1
        count += 1
    vidcap.release()
    return output_paths

# --- Landmark Detection ---
def analyze_frames_for_landmarks(frame_paths: List[str]) -> List[dict]:
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=True)
    pose = mp_pose.Pose(static_image_mode=True)

    results = []
    for image_path in frame_paths:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hand_result = hands.process(image_rgb)
        pose_result = pose.process(image_rgb)

        results.append({
            "frame": os.path.basename(image_path),
            "hand_landmarks": str(hand_result.multi_hand_landmarks),
            "pose_landmarks": str(pose_result.pose_landmarks)
        })
    return results

# --- Dummy Gloss Generator ---
def generate_dummy_gloss_sequence(landmark_data: List[dict]) -> List[str]:
    return ["IX-1", "WANT", "GO", "STORE"]

# --- Translate Gloss ---
def translate_gloss_to_english(glosses: List[str]) -> str:
    prompt = f"Translate this ASL gloss to natural English: {' '.join(glosses)}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# --- Streamlit UI ---
st.set_page_config(page_title="ASL to English Translator", layout="centered")
st.title("🤟 ASL Video to English Translator")

uploaded_file = st.file_uploader("Upload an ASL video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    with st.spinner("Extracting frames..."):
        frame_paths = extract_frames(temp_video_path, FRAMES_PER_SECOND)

    with st.spinner("Analyzing frames for landmarks..."):
        landmark_data = analyze_frames_for_landmarks(frame_paths)

    with st.spinner("Generating gloss sequence..."):
        glosses = generate_dummy_gloss_sequence(landmark_data)
        st.json({"Glosses": glosses})

    with st.spinner("Translating with ChatGPT..."):
        translation = translate_gloss_to_english(glosses)
        st.success("Translation complete!")
        st.markdown(f"### 📝 Translation\n{translation}")
