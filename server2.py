from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
import base64
import logging
import joblib
import xgboost as xgb
import atexit
from sklearn.base import BaseEstimator, ClassifierMixin
import subprocess

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# --- XGBoost Compatibility Wrapper ---
class XGBoostCompatWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# Load model artifacts
scaler = joblib.load('model_config/scaler_relative.pkl')
le = joblib.load('model_config/label_encoder_relative.pkl')
model = XGBoostCompatWrapper('model_config/asl_model_relative.json')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # Allow multiple hands
    min_detection_confidence=0.65
)

# Function to process landmarks
def process_landmarks(landmarks):
    landmarks_array = np.array(landmarks).reshape(1, 21, 3)
    wrist = landmarks_array[:, 0, :]
    relative = landmarks_array - wrist
    return relative[:, 1:, :].reshape(1, 60)

# Function to process frame and return predictions + annotated image
def process_frame(img):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            logging.warning("No hand detected in image.")
            return "no hand detected", None
        
        annotated_img = img.copy()
        predictions = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            raw_landmarks = []
            for landmark in hand_landmarks.landmark:
                raw_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            if len(raw_landmarks) != 63:
                logging.warning("Incomplete landmarks detected.")
                continue
            
            processed = process_landmarks(raw_landmarks)
            scaled = scaler.transform(processed)
            proba = model.predict_proba(scaled)
            gesture = le.inverse_transform([np.argmax(proba)])[0]
            predictions.append(gesture)
            
            # Draw annotations
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_img, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
        
        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return predictions, annotated_base64
    except Exception as e:
        logging.error(f"Error in process_frame: {e}")
        return "processing failed", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data.startswith('data:image/jpeg;base64,'):
            return jsonify({'error': 'Invalid image data'}), 400

        # Decode image
        base64_str = image_data.split(',')[1]
        image_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Process image
        results, annotated = process_frame(img)
        
        if not results or results == "processing failed":
            return jsonify({'error': 'Processing failed'}), 500
        
        return jsonify({'result': results, 'annotated': annotated})

    except Exception as e:
        logging.exception("Detection error:")
        return jsonify({'error': str(e)}), 500

@app.route("/page1")
def page1():
    return render_template("page1.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
