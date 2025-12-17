# ASL Hand Sign Recognition

An American Sign Language (ASL) hand sign recognition system built using MediaPipe for hand landmark extraction and XGBoost for gesture classification. The project includes a backend inference server, a browser-based frontend, and a video-to-text transcription pipeline.

This repository serves as a record of an end-to-end applied machine learning project.

---

## Overview

The system performs hand sign recognition using the following pipeline:

1. Hand detection and landmark extraction using MediaPipe Hands
2. Landmark preprocessing for translation invariance
3. Feature scaling
4. Gesture classification using an XGBoost model
5. Inference via a Flask-based backend

The project supports real-time webcam inference, image-based prediction, and offline video transcription.

---

## Project Structure

asl-hand-sign-recognition/  
├── data/  
│ └── datasets/ # Not tracked (linked below)  
│  
├── codebase/  
│ ├── model_config/  
│ │ ├── asl_model_relative.json  
│ │ ├── label_encoder_relative.pkl  
│ │ └── scaler_relative.pkl  
│ │  
│ ├── static/  
│ │ ├── css/  
│ │ └── js/  
│ │ └── script.js  
│ │  
│ ├── templates/  
│ │ └── HTML templates  
│ │  
│ ├── server2.py # Flask inference server  
│ └── transcribe.py # Video-to-text transcription  
│  
├── requirements.txt  
├── README.md  
├── .gitignore  
└── LICENSE


---

## Supported Classes

- Alphabets: A–Z
- Words:
  - apple
  - can
  - get
  - good
  - have
  - help
  - how
  - like
  - love
  - my
  - no
  - sorry
  - thankyou
  - want
  - yes
  - you
  - your
  - space

---

## Model and Preprocessing

- Hand landmarks are extracted using MediaPipe Hands (21 landmarks per hand).
- Each landmark consists of (x, y, z) coordinates.
- The wrist landmark is used as the origin to achieve translation invariance.
- The wrist is excluded from the final feature vector.
- Final input feature size: 60 dimensions.
- Features are standardized using a trained scaler.
- Classification is performed using an XGBoost model trained with a multi-class objective.

---

## Datasets

The datasets used for training are not included in this repository.

- Kaggle ASL Alphabet Dataset:  
  https://www.kaggle.com/datasets/grassknoted/asl-alphabet

- Roboflow ASL Dataset:  
  https://universe.roboflow.com/asl-dataset/asl-dataset-p9yw8/

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Inference Server

```bash
cd codebase
python server2.py
```

The server runs on `http://localhost:5000`

### Web Interface

- Open the application in a browser
- Allow webcam access
- The detected gesture and annotated output are returned by the backend

### Video Transcription (CLI)
```bash
python transcribe.py path/to/video.mp4
```

#### Transcription pipeline

- Extracts frames from the video
- Performs per-frame gesture prediction
- Applies sliding-window smoothing
- Removes consecutive duplicates and noise
- Outputs a cleaned word sequence and final transcription
- Optionally send duplicates to a LLM to smooth further and form a sentence 
Note: Gemini api is used in this project for smoothing, you need an API key to access it. Find out more about Gemini api here:
https://ai.google.dev/gemini-api/docs/api-key

## Technologies used

- Python
- MediaPipe
- OpenCV
- XGBoost
- scikit-learn
- Flask
- HTML, CSS, JavaScript
- FFmpeg

## Acknowledgements

- Google MediaPipe
- Kaggle dataset contributors
- Roboflow
## License

This project is licensed under the MIT License.
