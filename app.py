import os
import numpy as np
import librosa
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from pydub import AudioSegment
import tempfile
import base64 
# --- Page Configuration ---
st.set_page_config(page_title="குறள் வானி - Tamil Slang Detector", page_icon="🎤", layout="centered")

# --- Custom CSS for Enhanced UI ---
st.markdown("""
    <style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Tamil:wght@400;600;700&family=Montserrat:wght@300;400;600;700&display=swap');

    :root {
        --primary-color: #6a11cb;
        --secondary-color: #2575fc;
        --accent-color: #ff9a9e;
        --text-color: #f0f2f6;
        --background-start: #1a2a6c; /* Darker blue */
        --background-mid: #b21f1f;   /* Reddish */
        --background-end: #fdbb2d;   /* Yellow */
    }

    /* Animated gradient background */
    @keyframes gradientBG {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    .stApp {
        min-height: 100vh;
        background: linear-gradient(-45deg, var(--background-start), var(--background-mid), var(--background-end));
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite; /* Slower animation */
        color: var(--text-color);
        font-family: 'Montserrat', sans-serif;
    }

    /* Streamlit container customization */
    .css-h5fmus { /* This targets the main content block, common for Streamlit */
        background-color: rgba(0, 0, 0, 0.4); /* Semi-transparent dark overlay for readability */
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Logo Image Styling */
    .logo-img {
        display: block; /* Ensures it takes its own line */
        margin: 0 auto 2rem auto; /* Centers the image and adds space below */
        max-width: 200px; /* Adjust size as needed */
        height: auto;
        filter: drop-shadow(0 0 15px rgba(255, 255, 255, 0.5)); /* Adds a glow effect */
    }


    /* Tamil Title Styling */
    .title-tamil {
        font-family: 'IBM Plex Sans Tamil', sans-serif;
        font-size: 4.5rem; /* Larger font size */
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.8);
        background: -webkit-linear-gradient(45deg, #FFD700, #FFA500); /* Gold to Orange gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .subtitle-english {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8rem;
        text-align: center;
        color: #f0f2f6;
        margin-top: 0;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
    }

    /* Section Headers */
    h2 {
        font-family: 'Montserrat', sans-serif;
        color: var(--accent-color);
        text-align: center;
        font-size: 2.2rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    }

    /* General text styling */
    p {
        font-family: 'Montserrat', sans-serif;
        color: var(--text-color);
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* Make Streamlit buttons prettier */
    .stButton>button {
        background: linear-gradient(45deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        font-weight: 600;
        border-radius: 15px; /* More rounded */
        padding: 0.8rem 1.8rem; /* Larger padding */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: none;
        font-size: 1.1rem;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, var(--secondary-color) 0%, var(--primary-color) 100%);
        transform: translateY(-3px); /* Lift effect on hover */
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.6);
        cursor: pointer;
    }

    /* File uploader styling */
    .stFileUploader label {
        color: var(--text-color);
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.4);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: var(--accent-color);
    }
    .stFileUploader p {
        color: var(--text-color);
        font-size: 1.1rem;
    }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(90deg, #155724, #007bff); /* Green to Blue gradient */
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    .prediction-box h4 {
        color: #e0ffe0; /* Lighter green */
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .prediction-box p {
        color: #f0f2f6; /* Changed to match general text color for clarity */
        font-size: 1.2rem;
    }

    /* Confidence text within prediction box */
    .confidence-text {
        color: #FFD700; /* Gold color for confidence to stand out */
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    .stAlert {
        border-radius: 10px;
        background-color: rgba(255, 99, 71, 0.7); /* Tomato red for error */
        color: black;
        font-weight: 600;
    }

    </style>
""", unsafe_allow_html=True)

# --- Logo Insertion ---
# Ensure your 'logo.png' is in the same directory as this script
try:
    with open("logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <img src="data:image/png;base64,{logo_base64}" class="logo-img">
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("Logo file 'logo.png' not found. Please ensure it's in the same directory.")
except Exception as e:
    st.error(f"An error occurred while loading the logo: {e}")

# --- App Title and Description ---
st.markdown("""
    <h1 class="title-tamil">குறள் வானி</h1>
    <p class="subtitle-english">Tamil Slang Detection App</p>
    <p style="text-align: center; color: var(--text-color); font-size: 1.2rem; margin-bottom: 3rem;">
        Upload an audio file (WAV/MP3) and let our AI determine its Tamil slang category.
    </p>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model_and_encoder():
    model = load_model("tamil_slang_model.h5")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model_and_encoder()

SAMPLE_RATE = 22050
MFCC_FEATURES = 40

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)
        if mfcc.shape[1] < 40:
            return None
        return mfcc[:, :40]
    except:
        return None

uploaded = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        if uploaded.name.endswith(".mp3"):
            audio = AudioSegment.from_file(uploaded, format="mp3")
            audio.export(tmp.name, format="wav")
        else:
            tmp.write(uploaded.read())

        feat = extract_features(tmp.name)
        if feat is None:
            st.error("⚠️ Invalid or too short audio. Try another.")
        else:
            feat = np.expand_dims(feat, axis=[0, -1])
            preds = model.predict(feat)[0]
            label = le.inverse_transform([np.argmax(preds)])[0]
            confidence = np.max(preds) * 100

            st.markdown(f"""
                <div style="background-color:#d4edda;padding:1rem;border-radius:10px;">
                    <h4 style='color:#155724;'>🎯 Prediction: <b>{label.title()}</b></h4>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
