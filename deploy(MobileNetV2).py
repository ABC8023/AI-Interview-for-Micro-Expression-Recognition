from zipfile import ZipFile
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import logging
from werkzeug.utils import secure_filename
import subprocess
from uuid import uuid4
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configuration
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
UPLOAD_FOLDER = STATIC_DIR / 'uploads'
PROCESSED_FOLDER = STATIC_DIR / 'processed'
MODEL_PATH = "mobilenet_micro_expression_classifier.keras"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Emotion classes
EMOTIONS = ["disgust", "fear", "happiness", "repression", "sadness", "surprise"]

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure directories exist
for folder in (UPLOAD_FOLDER, PROCESSED_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load ML model
def load_model():
    """Load the pre-trained MobileNetV2 model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_video_writer(output_path: Path, frame, fps: float):
    """Create a video writer with appropriate codec"""
    height, width = frame.shape[:2]
    width, height = width - (width % 2), height - (height % 2)
    
    # Try codecs in order of preference
    for codec in ['mp4v', 'XVID', 'MJPG']:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height)
        )
        if writer.isOpened():
            return writer
    return None

def convert_video(input_path: Path, output_path: Path):
    """Convert video to MP4 using FFmpeg for better compatibility"""
    try:
        cmd = [
            r'C:\ffmpeg\bin\ffmpeg.exe', '-y',
            '-i', str(input_path),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22',
            '-c:a', 'aac',
            '-strict', 'experimental',
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Video converted: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg conversion failed: {e.stderr.decode()}")
        return False

def analyze_frame(frame, model):
    """Analyze a single frame for micro-expressions"""
    try:
        # Preprocess frame
        resized = cv2.resize(frame, (128, 128)) / 255.0
        # Predict emotions
        probs = model.predict(np.expand_dims(resized, 0), verbose=0)[0]
        emotion_idx = np.argmax(probs)
        confidence = probs[emotion_idx]
        return EMOTIONS[emotion_idx], confidence
    except Exception as e:
        logger.error(f"Frame analysis error: {str(e)}")
        return "unknown", 0.0

def process_video(input_path: Path, output_path: Path):
    """Process video to detect micro-expressions"""
    if model is None:
        return None, None, None

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"❌ Could not open video: {input_path}")
        return None, None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame = cap.read()
    if not ret:
        logger.error("❌ No frames in video")
        return None, None, None

    # Initialize video writer
    writer = create_video_writer(output_path, frame, fps)
    if not writer:
        logger.error("❌ Could not create video writer")
        return None, None, None

    # Initialize counters
    emotion_counts = {e: 0 for e in EMOTIONS}
    total_frames = 0

    # Process frames
    while ret:
        emotion, confidence = analyze_frame(frame, model)
        
        # Update counts
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        
        # Annotate frame
        label = f"{emotion} ({confidence:.1%})"
        cv2.putText(frame, label, (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        writer.write(frame)
        total_frames += 1
        ret, frame = cap.read()

    # Release resources
    cap.release()
    writer.release()

    # --- Emotion Group Normalization ---
    positive_emotions = ["happiness", "surprise"]
    negative_emotions = ["disgust", "fear", "repression", "sadness"]

    # Normalize each emotion's contribution by group size
    normalized_positive = sum(emotion_counts[e] / len(positive_emotions) for e in positive_emotions)
    normalized_negative = sum(emotion_counts[e] / len(negative_emotions) for e in negative_emotions)

    # --- Final Positivity Score (Balanced 50/50) ---
    total_normalized = normalized_positive + normalized_negative
    if total_normalized > 0:
        positivity_score = ((normalized_positive - normalized_negative) / total_normalized) * 100
    else:
        positivity_score = 0.0

    # --- Final Stress Score (Balanced 50/50) ---
    stress_score = (normalized_negative / total_normalized) * 100 if total_normalized > 0 else 0.0

    # Re-encode for web compatibility
    web_output = output_path.with_name(f"web_{output_path.name}")
    if convert_video(output_path, web_output):
        return web_output, positivity_score, stress_score, emotion_counts
    # Fallback if conversion failed: still return all four values
    return output_path, positivity_score, stress_score, emotion_counts

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

# Also make sure this import is at the top:
from flask import send_from_directory

def index():
    """Serve the main interface"""
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        # Save original file
        file_id = uuid4().hex
        original_ext = Path(file.filename).suffix.lower()
        original_path = UPLOAD_FOLDER / f"{file_id}{original_ext}"
        file.save(original_path)
        logger.info(f"Uploaded: {original_path.name}")

        # Process video
        output_name = f"processed_{file_id}.mp4"
        result_path, positivity_score, stress_score, counts = process_video(
            original_path,
            PROCESSED_FOLDER / output_name
        )

        if not result_path or not result_path.exists():
            logger.error("Processed video file not found or not created")
            return jsonify({"error": "Video processing failed"}), 500

        # --- NEW: Build results text file ---
        txt_path = PROCESSED_FOLDER / f"results_{file_id}.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"File ID: {file_id}\n")
                f.write(f"Processed Video: {result_path.name}\n\n")
                f.write(f"Positivity Score: {round(positivity_score, 2)}%\n")
                f.write(f"Stress Score: {round(stress_score, 2)}%\n\n")
                f.write("Emotion Counts:\n")
                for emo in EMOTIONS:
                    f.write(f"  - {emo}: {counts.get(emo, 0)}\n")
        except Exception as e:
            logger.error(f"Failed to write results text: {e}")

        # --- NEW: Zip video + text together ---
        zip_name = f"analysis_{file_id}.zip"
        zip_path = PROCESSED_FOLDER / zip_name
        try:
            with ZipFile(zip_path, "w") as z:
                z.write(result_path, arcname="processed_video.mp4")
                z.write(txt_path,   arcname="results.txt")
        except Exception as e:
            logger.error(f"Failed to create zip: {e}")

        # Return JSON (keep existing fields, add zip_url)
        return jsonify({
            "video_url": f"/processed/{result_path.name}",
            "zip_url": f"/processed/{zip_name}",     # NEW
            "positivity_score": round(positivity_score, 2),
            "stress_score": round(stress_score, 2),
            "emotion_counts": counts,
            "file_id": file_id
        })
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": "Server error processing video"}), 500

@app.route('/processed/<filename>')
def serve_processed(filename):
    """Serve processed video files"""
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)