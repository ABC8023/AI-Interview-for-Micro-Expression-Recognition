# 🧠 AI Interview for Micro-Expression Recognition

This project is an AI-based interview assistant that analyzes subtle facial micro-expressions from uploaded interview videos using a MobileNetV2 deep learning model.
It provides feedback such as Emotional Positivity Score and Stress/Discomfort Score, helping users or recruiters gain deeper emotional insights during interviews.

## 🚀 Features

⦁	Upload an interview video through a simple Flask web interface.
⦁	Detects micro-expressions using a MobileNetV2-based classifier trained on CASME II and DFME datasets.
⦁	Overlays top detected emotions (with confidence %) on the processed video.
⦁	Automatically generates:
  ⦁ 🎥 Processed video with emotion labels
  ⦁	📊 Emotional Positivity and Stress/Discomfort Scores
  ⦁	📁 Downloadable ZIP (includes both video + result summary)

## 🧩 Project Structure
FYP Final/

│  deploy(MobileNetV2).py          # Flask app – runs server, processes uploaded videos
│  MobileNetV2.py                  # Model training & evaluation (TensorFlow MobileNetV2)
│  Video_Preprocess(CASME).py      # Dataset preprocessing (CASME II)
│  Video_Preprocess(DFME).py       # Dataset preprocessing (DFME)
│  mobilenet_micro_expression_classifier.keras  # Pre-trained model file
│  index.html                      # Web UI for video upload & analysis
│
└─ static/
   ├─ uploads/                     # Automatically created folder for raw uploads
   └─ processed/                   # Automatically created folder for results

## 🛠️ Requirements

⦁	Windows 10/11 (64-bit)
⦁	Python 3.8 (recommended for TensorFlow 2.10)
⦁	FFmpeg (for MP4 video conversion)
⦁	Internet connection (for initial dependency install)

## ⚙️ Installation & Setup (Windows + VS Code)
1️⃣ Open Terminal in VS Code inside your project folder:

cd "C:\Users\User\FYP Final"

2️⃣ Create Virtual Environment
C:\Users\User\AppData\Local\Programs\Python\Python38\python.exe -m venv cbs_fyp
cbs_fyp\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Trained Model

Place your trained model file here:

FYP Final\mobilenet_micro_expression_classifier.keras

(Or update the MODEL_PATH variable in deploy(MobileNetV2).py.)

5️⃣ Install FFmpeg

Unzip or install FFmpeg so that:

C:\ffmpeg\bin\ffmpeg.exe


Check installation:

ffmpeg -version

6️⃣ Run the App
python "deploy(MobileNetV2).py"

You should see:

Model loaded successfully
 * Running on http://127.0.0.1:5000/

7️⃣ Use the Web Interface

Open your browser and visit:

http://127.0.0.1:5000

Upload an interview video (MP4/AVI/MOV/MKV, ≤100 MB).
After processing, download your results as a ZIP file containing:

⦁	processed_video.mp4 — video with detected emotions
⦁	results.txt — detailed scores and emotion breakdown

## 🧮 How It Works

The video is split into frames.

Each frame is analyzed using MobileNetV2, which predicts one of six emotions:

Disgust, Fear, Happiness, Repression, Sadness, Surprise

Emotion counts are aggregated and normalized.

Final results:

Emotional Positivity Score → higher = more positive emotions

Stress/Discomfort Score → higher = more negative emotions

## 🧰 Tech Stack

Python, TensorFlow/Keras, OpenCV, Flask

MobileNetV2 for transfer learning

FFmpeg for video encoding/decoding

CASME II / DFME datasets for training

## 📈 Example Output
Positivity Score: 68.4%
Stress Score: 31.6%

Emotion Counts:
 - happiness: 230
 - surprise: 145
 - sadness: 40
 - repression: 28
 - fear: 22
 - disgust: 18

## 🧑‍💻 Author
Chin Bao Sheng
Bachelor’s Final Year Project – AI Interview for Micro-Expression Recognition
