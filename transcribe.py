#!/usr/bin/env python3
import os
import cv2
import glob
import shutil
import subprocess
import tempfile
import numpy as np
import mediapipe as mp
import joblib
import logging
from statistics import mode
from collections import deque
from typing import List
import google.generativeai as genai

# -------------------------------
# Custom XGBoost Model Wrapper
# -------------------------------
import xgboost as xgb

class XGBoostCompatWrapper:
    def __init__(self, json_model_path: str):
        self.booster = xgb.Booster()
        self.booster.load_model(json_model_path)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(X)
        # Assuming model was trained with multi:softprob
        proba = self.booster.predict(dmatrix)
        # proba shape: (n_samples, n_classes)
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# -------------------------------
# Load Model Artifacts
# -------------------------------
import argparse

# Update these paths as needed.
SCALER_PATH = 'model_config/scaler_relative.pkl'
LABEL_ENCODER_PATH = 'model_config/label_encoder_relative.pkl'
MODEL_JSON_PATH = 'model_config/asl_model_relative.json'

scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
model = XGBoostCompatWrapper(MODEL_JSON_PATH)

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# -------------------------------
# Frame Processing Functions
# -------------------------------
def process_landmarks(landmarks: list) -> np.ndarray:
    """Convert raw 63-element landmarks to a 60-element model input (relative coordinates)."""
    # Reshape to (1, 21, 3)
    landmarks_array = np.array(landmarks).reshape(1, 21, 3)
    # Use landmark 0 (wrist) as origin.
    wrist = landmarks_array[:, 0, :]
    relative = landmarks_array - wrist
    # Drop the wrist landmark (first three values)
    processed = relative[:, 1:, :].reshape(1, 60)
    return processed

def process_frame(img) -> str:
    """Process a single frame: extract landmarks, preprocess, predict label."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return "no hand detected"
    
    try:
        raw_landmarks = []
        # Use first detected hand
        for landmark in results.multi_hand_landmarks[0].landmark:
            raw_landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        if len(raw_landmarks) != 63:
            return "no hand detected"
        
        processed = process_landmarks(raw_landmarks)
        scaled = scaler.transform(processed)
        proba = model.predict_proba(scaled)
        predicted = le.inverse_transform([np.argmax(proba)])[0]
        return predicted
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return "no hand detected"

# -------------------------------
# Smoothing / Noise Imputation
# -------------------------------
def smooth_predictions(preds: List[str], window_size: int = 5) -> List[str]:
    """
    Applies a sliding-window majority filter and then collapses consecutive duplicates.
    Ignores "no hand detected" predictions.
    """
    if not preds:
        return []
    
    # Replace "no hand detected" with None and filter later.
    preds_filtered = [p if p != "no hand detected" else None for p in preds]
    n = len(preds_filtered)
    smoothed = []
    
    # Use a sliding window majority vote.
    for i in range(n):
        window = preds_filtered[max(0, i - window_size//2) : min(n, i + window_size//2 + 1)]
        # Only consider valid predictions in the window.
        valid = [p for p in window if p is not None]
        if valid:
            try:
                majority = mode(valid)
            except Exception:
                # In case of tie, choose the current value if valid.
                majority = preds_filtered[i] if preds_filtered[i] is not None else valid[0]
            smoothed.append(majority)
        else:
            smoothed.append(None)
    
    # Remove consecutive duplicates and Nones.
    final = []
    for pred in smoothed:
        if pred is None:
            continue
        if not final or pred != final[-1]:
            final.append(pred)
    
    return final

# -------------------------------
# Dummy analyse() Function
# -------------------------------
def analyse(word_list: List[str]) -> str:
    """Generates text using the Gemini API. Used to filter noise and get meaningful sentence

    Args:
        word_list: List of detected words

    Returns:
        The generated text response, or None if an error occurs.
    """
    genai.configure(api_key=str(open('api_key.txt','r').read()))
    # Create the Model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config
    )
    chat = model.start_chat(history=[])
    response = chat.send_message(f"I will give you a list of words. It has some noise words and an actual meaningful sentence. Identify the sentense and filter the noise. Return only the statement you identified. word list:{word_list}")
    return response.text
    
# -------------------------------
# FFmpeg Frame Extraction
# -------------------------------
def extract_frames(video_path: str, output_dir: str) -> None:
    """
    Uses FFmpeg to extract all frames from the video into output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # FFmpeg command: extract frames as jpg images.
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-qscale:v", "2",  # high quality
        os.path.join(output_dir, "frame_%06d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# -------------------------------
# Main Function
# -------------------------------
def main(video_path: str):
    # Create a temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix="asl_frames_")
    print(f"Extracting frames to {temp_dir} ...")
    
    try:
        extract_frames(video_path, temp_dir)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return
    
    # Get sorted list of frame file paths
    frame_files = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
    if not frame_files:
        print("No frames extracted.")
        shutil.rmtree(temp_dir)
        return
    
    predictions = []
    print("Processing frames...")
    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        if img is None:
            continue
        pred = process_frame(img)
        predictions.append(pred)
    
    # Clean up temporary frames
    shutil.rmtree(temp_dir)
    
    # Smooth predictions and remove noise
    final_words = smooth_predictions(predictions, window_size=5)
    
    # Pass the processed list to analyse() to obtain the final sentence.
    final_sentence = analyse(final_words)
    
    # Output results.
    print("Processed word list:")
    print(final_words)
    print("\nFinal transcription:")
    print(final_sentence)

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe ASL video to text.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    args = parser.parse_args()
    
    main(args.video_path)
    
    # Release MediaPipe resources.
    hands.close()