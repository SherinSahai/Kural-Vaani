# 🎙️ Kural Vaani - Tamil Slang Detection App

**Kural Vaani** is a deep learning-powered application that detects and classifies different types of spoken Tamil slangs from audio samples. The app is built using Convolutional Neural Networks (CNN) trained on MFCC features extracted from real-world Tamil dialect datasets.

## 🔍 Project Overview

Tamil, being a classical and regionally diverse language, has a wide range of spoken dialects and slang variations. Kural Vaani identifies five major categories:

- **Madurai Slang**
- **Chennai Slang**
- **Thirunelveli Slang**
- **Sri Lankan Tamil Slang**
- **Standard Tamil** (without slang)

This system helps in regional dialect analysis, audio-based language processing, and sociolinguistic research applications.

---

## 🗂️ Dataset Details

### 🎧 Audio Collection

Audio samples were **manually collected and categorized** based on regional slang types:

1. **Madurai Slang**
   - Collected from native speakers in Madurai and surrounding southern districts.
   - Informal street-level phrases and strong regional accents.

2. **Chennai Slang**
   - Collected from urban Chennai speech samples.
   - Includes popular colloquial phrases used in day-to-day city life.

3. **Thirunelveli Slang**
   - Rich in distinct phonetic patterns and vocabulary.
   - Sourced from conversations of native speakers from Tirunelveli and Tuticorin districts.

4. **Sri Lankan Tamil**
   - Unique pronunciation patterns and grammar.
   - Collected from Tamil-speaking communities in Jaffna and Batticaloa.

5. **Standard Tamil (Without Slang)**
   - Sourced from public speeches, Tamil news media, and textbook narration.
   - Serves as the control class for pure, slang-free Tamil.

### 🛠️ Preprocessing
- Audio was converted to WAV format.
- Extracted MFCC (Mel-Frequency Cepstral Coefficients) with 40 coefficients per sample.
- All audio samples were normalized and resized to a uniform shape (40x40) for CNN input.

---

## 🧠 Model Architecture

A Convolutional Neural Network (CNN) was used with the following layers:
- 2D Convolution + MaxPooling
- Dropout layers to prevent overfitting
- Dense layer with Softmax activation for 5-class classification

Model trained using:
- **Categorical Crossentropy Loss**
- **Adam Optimizer**
- **Accuracy** as the primary metric

---

## 🚀 Features

- 🎤 Upload your own audio file to detect its Tamil slang.
- 📊 Real-time prediction with confidence score.
- 🔍 Slang classification in under 2 seconds.

---

## 📦 Technology Stack

- Python
- TensorFlow / Keras
- Librosa
- NumPy, Pandas
- Streamlit (for the frontend)
- Joblib (for label encoding)

---

