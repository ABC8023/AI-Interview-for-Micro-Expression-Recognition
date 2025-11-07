import os
import shutil
import pandas as pd

# ─────── Configuration ───────
TRAIN_DATA_PATH = "D:/ChinWY/train_data/train_data"
EXCEL_FILE_PATH = "C:/Users/PC/Downloads/full_release_annotation/full_release_annotation/training_full_with_AU.xlsx"
OUTPUT_FOLDER = "C:/Users/PC/CBS_FYP2/train_data_categorized"

# ─────── Load Excel Data ───────
df = pd.read_excel(EXCEL_FILE_PATH)
df["Filename"] = df["Filename"].astype(str).str.strip()
df["Emotion"] = df["Emotion"].str.lower().str.strip().str.replace(" ", "_")

# ─────── Create Output Directory ───────
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────── Copy Images Based on Emotion ───────
for idx, row in df.iterrows():
    folder_name = row["Filename"]  # e.g., sub0003_01
    emotion = row["Emotion"]

    src_folder = os.path.join(TRAIN_DATA_PATH, folder_name)
    dest_folder = os.path.join(OUTPUT_FOLDER, emotion)

    if not os.path.exists(src_folder):
        print(f"⚠️ Warning: Source folder does not exist -> {src_folder}")
        continue

    os.makedirs(dest_folder, exist_ok=True)

    for img_file in os.listdir(src_folder):
        src_img_path = os.path.join(src_folder, img_file)
        dest_img_path = os.path.join(dest_folder, f"{folder_name}_{img_file}")
        shutil.copy2(src_img_path, dest_img_path)

print("✅ Categorization complete.")