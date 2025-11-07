# 1. Put all frames from CASME2_RAW_selected to CASME2_by_emotion
import os
import shutil
import pandas as pd

# Define paths
DATASET_PATH = "C:\\Users\\PC\\CASME2_RAW_selected\\CASME2_RAW_selected"
EXCEL_FILE_PATH = "C:\\Users\\PC\\CASME2-coding-20140508.xlsx"
OUTPUT_PATH = "CASME2_by_emotion"

# Load the metadatame
df = pd.read_excel(EXCEL_FILE_PATH, sheet_name="Sheet1")

# Create the output root directory if not exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Loop through each row in the Excel sheet
for index, row in df.iterrows():
    subject = f"sub{int(row['Subject']):02d}"
    filename = str(row['Filename'])
    emotion = str(row['Estimated Emotion']).strip().lower().replace(" ", "_")

    # Full path to the expression folder
    expression_path = os.path.join(DATASET_PATH, subject, filename)
    if not os.path.exists(expression_path):
        print(f"Warning: Path not found -> {expression_path}")
        continue

    # Destination emotion folder
    emotion_folder = os.path.join(OUTPUT_PATH, emotion)
    os.makedirs(emotion_folder, exist_ok=True)

    # Copy all images from the expression folder to the emotion folder
    for img_file in os.listdir(expression_path):
        if img_file.endswith(".jpg"):
            src_path = os.path.join(expression_path, img_file)
            new_filename = f"{subject}_{filename}_{img_file}"
            dst_path = os.path.join(emotion_folder, new_filename)
            shutil.copy2(src_path, dst_path)

print(f"✅ All images have been categorized into: {OUTPUT_PATH}")

# Count images per emotion
emotion_counts = {}
for emotion in os.listdir(OUTPUT_PATH):
    emotion_folder = os.path.join(OUTPUT_PATH, emotion)
    if os.path.isdir(emotion_folder):
        count = len([f for f in os.listdir(emotion_folder) if f.endswith('.jpg')])
        emotion_counts[emotion] = count

# Print the results
print("📊 Image count per emotion class:")
for emotion, count in sorted(emotion_counts.items()):
    print(f"{emotion:20}: {count}")


# 2. Remove ApexFrame and the "others" folder from CASME2_by_emotion folder
import os
import shutil
import pandas as pd
import re

# Paths
EXCEL_FILE_PATH = "C:\\Users\\PC\\CASME2-coding-20140508.xlsx"
INPUT_PATH = "CASME2_by_emotion"
OUTPUT_PATH = "CASME2_by_emotion_filtered"
APEX_PATH = "CASME2_apex_frames"

# Load metadata
df = pd.read_excel(EXCEL_FILE_PATH, sheet_name="Sheet1")

# Helper to clean ApexFrame value
def clean_frame_value(value):
    if isinstance(value, str):
        value = value.replace("/", "").strip()
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

# Build a lookup dictionary for ApexFrame
apex_lookup = {}
for _, row in df.iterrows():
    subject = f"sub{int(row['Subject']):02d}"
    filename = str(row['Filename'])
    apex = clean_frame_value(row['ApexFrame'])
    apex_lookup[f"{subject}_{filename}"] = apex

# Create output paths
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(APEX_PATH, exist_ok=True)

# Process emotion folders
for emotion in os.listdir(INPUT_PATH):
    if emotion.lower() == "others":
        print(f"Skipping 'others' folder.")
        continue

    input_emotion_path = os.path.join(INPUT_PATH, emotion)
    output_emotion_path = os.path.join(OUTPUT_PATH, emotion)
    apex_emotion_path = os.path.join(APEX_PATH, emotion)

    os.makedirs(output_emotion_path, exist_ok=True)
    os.makedirs(apex_emotion_path, exist_ok=True)

    for file in os.listdir(input_emotion_path):
        if not file.endswith(".jpg"):
            continue

        match = re.match(r"(sub\d+)_([A-Z]+\d+_?\d*f?)_img(\d+)\.jpg", file)
        if not match:
            print(f"Skipping unrecognized file format: {file}")
            continue

        subject, filename, frame_num = match.groups()
        key = f"{subject}_{filename}"
        apex_frame = apex_lookup.get(key)

        src_path = os.path.join(input_emotion_path, file)

        if apex_frame is not None and int(frame_num) == int(apex_frame):
            # Save to apex folder
            dst_path = os.path.join(apex_emotion_path, file)
            shutil.copy2(src_path, dst_path)
            print(f"Saved ApexFrame: {file}")
        else:
            # Save to normal filtered folder
            dst_path = os.path.join(output_emotion_path, file)
            shutil.copy2(src_path, dst_path)

print(f"\n✅ Onset/Offset images saved to '{OUTPUT_PATH}'")
print(f"✅ Apex frames saved separately to '{APEX_PATH}'")


# Count images per emotion
emotion_counts = {}
for emotion in os.listdir(OUTPUT_PATH):
    emotion_folder = os.path.join(OUTPUT_PATH, emotion)
    if os.path.isdir(emotion_folder):
        count = len([f for f in os.listdir(emotion_folder) if f.endswith('.jpg')])
        emotion_counts[emotion] = count

# Print counts
print("📊 Image count per emotion class:")
for emotion, count in sorted(emotion_counts.items()):
    print(f"{emotion:20}: {count}")

# Count images per emotion
apex_counts = {}
for emotion in os.listdir(APEX_PATH):
    emotion_folder = os.path.join(APEX_PATH, emotion)
    if os.path.isdir(emotion_folder):
        count = len([f for f in os.listdir(emotion_folder) if f.endswith('.jpg')])
        apex_counts[emotion] = count

# Print the results
print("📌 ApexFrame image count per emotion class:")
for emotion, count in sorted(apex_counts.items()):
    print(f"{emotion:20}: {count}")

# 4. Augment each image by a fixed number of times (e.g. 10 augmentations per image)
import os
import numpy as np
import shutil
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import torch

INPUT_PATH = "CASME2_by_emotion_filtered"
OUTPUT_PATH = "CASME2_augmented"
AUGMENTATIONS_PER_IMAGE = 10  # configurable

# Define augmentation pipeline
augmentor = ImageDataGenerator(
    rotation_range=20,        # Rotate images randomly within ±20 degrees
    width_shift_range=0.2,    # Shift images horizontally by up to 20% of width
    height_shift_range=0.2,   # Shift images vertically by up to 20% of height
    shear_range=0.2,          # Apply shear transformations (slanting effect)
    zoom_range=0.2,           # Zoom in or out by up to 20%
    horizontal_flip=True,     # Randomly flip images horizontally
    fill_mode='nearest'       # Fill in missing pixels after transformation using nearest pixel
)

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

for emotion in os.listdir(INPUT_PATH):
    input_emotion_path = os.path.join(INPUT_PATH, emotion)
    output_emotion_path = os.path.join(OUTPUT_PATH, emotion)
    os.makedirs(output_emotion_path, exist_ok=True)

    images = [f for f in os.listdir(input_emotion_path) if f.endswith('.jpg')]
    
    # Copy original images first
    for img_file in images:
        shutil.copy2(os.path.join(input_emotion_path, img_file), os.path.join(output_emotion_path, img_file))

    print(f"🔄 Augmenting '{emotion}' with {AUGMENTATIONS_PER_IMAGE} augmentations per image...")

    for img_file in tqdm(images):
        img_path = os.path.join(input_emotion_path, img_file)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        aug_gen = augmentor.flow(img_array, batch_size=1)
        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = next(aug_gen)[0].astype('uint8')
            aug_img_pil = Image.fromarray(aug_img)
            aug_filename = f"aug_{i}_{img_file}"
            save_path = os.path.join(output_emotion_path, aug_filename)
            aug_img_pil.save(save_path)

print(f"\n✅ Augmentation complete using {AUGMENTATIONS_PER_IMAGE} augmentations per image")

# 📊 Count and display the number of images per class
print("\n📁 Image count per class in augmented dataset:")
for emotion in sorted(os.listdir(OUTPUT_PATH)):
    emotion_path = os.path.join(OUTPUT_PATH, emotion)
    if os.path.isdir(emotion_path):
        num_images = len([f for f in os.listdir(emotion_path) if f.endswith('.jpg')])
        print(f"{emotion}: {num_images} images")



# 5. Merge Apex Frames Back Into Augmented Dataset
AUGMENTED_PATH = "CASME2_augmented"
APEX_PATH = "CASME2_apex_frames"
MERGED_PATH = "CASME2_augmented_with_apex_frames"

# Create merged folder
os.makedirs(MERGED_PATH, exist_ok=True)

# Step 1: Copy entire augmented dataset to the new folder
for emotion in os.listdir(AUGMENTED_PATH):
    src_emotion_path = os.path.join(AUGMENTED_PATH, emotion)
    dst_emotion_path = os.path.join(MERGED_PATH, emotion)
    os.makedirs(dst_emotion_path, exist_ok=True)

    for file in os.listdir(src_emotion_path):
        src = os.path.join(src_emotion_path, file)
        dst = os.path.join(dst_emotion_path, file)
        shutil.copy2(src, dst)

print("✅ Augmented dataset copied to CASME2_augmented_with_apex_frames")

# Step 2: Add apex frames to the merged dataset
for emotion in os.listdir(APEX_PATH):
    apex_emotion_path = os.path.join(APEX_PATH, emotion)
    target_emotion_path = os.path.join(MERGED_PATH, emotion)

    if not os.path.exists(target_emotion_path):
        os.makedirs(target_emotion_path, exist_ok=True)

    for file in os.listdir(apex_emotion_path):
        src = os.path.join(apex_emotion_path, file)
        dst = os.path.join(target_emotion_path, file)

        # Optional: prevent overwriting (in case names overlap, unlikely)
        if os.path.exists(dst):
            base, ext = os.path.splitext(file)
            file = f"{base}_apex{ext}"
            dst = os.path.join(target_emotion_path, file)

        shutil.copy2(src, dst)

print("✅ Apex frames added to CASME2_augmented_with_apex_frames")

# 🧮 Compute Class Weights for Cost-Sensitive Learning
from collections import Counter

# Count images per class from the final merged dataset
merged_counts = {
    emotion: len([f for f in os.listdir(os.path.join(MERGED_PATH, emotion)) if f.endswith('.jpg')])
    for emotion in os.listdir(MERGED_PATH)
}

# Print stats
print("\n📊 Final image counts after merging apex frames:")
for cls, count in merged_counts.items():
    print(f"{cls}: {count}")



# 6. Augment apex frames using a fixed number of augmentations per image (same as Step 4)
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Paths
INPUT_APEX_PATH = "CASME2_apex_frames"
OUTPUT_PATH = "CASME2_augmented_apex_frames_for_testing"
TRAINING_PATH = "CASME2_augmented_with_apex_frames"
AUGMENTATIONS_PER_IMAGE = 10  # Same as in Step 4

# Define augmentation pipeline (same as Step 4)
augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Step 1: Augment and remove original apex frames
for emotion in os.listdir(INPUT_APEX_PATH):
    emotion_folder = os.path.join(INPUT_APEX_PATH, emotion)
    if not os.path.isdir(emotion_folder):
        continue

    output_emotion_path = os.path.join(OUTPUT_PATH, emotion)
    os.makedirs(output_emotion_path, exist_ok=True)

    # Load apex images
    image_files = [f for f in os.listdir(emotion_folder) if f.endswith(".jpg")]

    print(f"🔄 Augmenting '{emotion}' apex images ({len(image_files)} originals × {AUGMENTATIONS_PER_IMAGE} augmentations each)")

    for img_file in tqdm(image_files):
        img_path = os.path.join(emotion_folder, img_file)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img).reshape((1,) + np.array(img).shape)

        aug_gen = augmentor.flow(img_array, batch_size=1)

        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = next(aug_gen)[0].astype('uint8')
            aug_img_pil = Image.fromarray(aug_img)
            new_filename = f"aug_{i}_{img_file}"
            save_path = os.path.join(output_emotion_path, new_filename)
            aug_img_pil.save(save_path)

print(f"\n✅ Apex images successfully augmented using {AUGMENTATIONS_PER_IMAGE}x per original.")
print(f"🗂️  Saved to: '{OUTPUT_PATH}'")

# Step 2: Show final image counts for each class
print("\n📊 Final image counts in testing set:")
testing_counts = {}
for emotion in os.listdir(OUTPUT_PATH):
    emotion_folder = os.path.join(OUTPUT_PATH, emotion)
    if not os.path.isdir(emotion_folder):
        continue

    count = len([f for f in os.listdir(emotion_folder) if f.endswith(".jpg")])
    testing_counts[emotion] = count
    print(f"{emotion}: {count} images")

# Step 3: Compare ratio with training set
print("\n📐 Test/Train Ratio per class:")
for emotion in sorted(testing_counts.keys()):
    test_count = testing_counts[emotion]
    training_folder = os.path.join(TRAINING_PATH, emotion)
    if not os.path.isdir(training_folder):
        print(f"{emotion}: ❌ Training data not found")
        continue

    train_count = len([f for f in os.listdir(training_folder) if f.endswith(".jpg")])
    ratio = test_count / train_count if train_count > 0 else 0
    print(f"{emotion}: {test_count} / {train_count} = {ratio:.2f}")




# Now you will get the updated training set and testing set
import os
import shutil
from collections import defaultdict
import random
import torch

TRAIN_PATH = "CASME2_augmented_with_apex_frames"
TEST_PATH = "CASME2_augmented_apex_frames_for_testing"
TARGET_RATIO = 0.2  # 20% testing

# Step 1: Load current image counts
def get_augmented_images(path):
    return [f for f in os.listdir(path) if f.endswith(".jpg") and f.startswith("aug_")]

training_counts = {}
testing_counts = {}

for emotion in os.listdir(TRAIN_PATH):
    train_emotion_path = os.path.join(TRAIN_PATH, emotion)
    test_emotion_path = os.path.join(TEST_PATH, emotion)

    train_aug_imgs = get_augmented_images(train_emotion_path)
    training_counts[emotion] = len(train_aug_imgs)

    if not os.path.exists(test_emotion_path):
        os.makedirs(test_emotion_path)
    test_imgs = get_augmented_images(test_emotion_path)
    testing_counts[emotion] = len(test_imgs)

# Step 2: Top-up process
for emotion in training_counts:
    current_train = training_counts[emotion]
    current_test = testing_counts.get(emotion, 0)
    total = current_train + current_test
    target_test = int(total * TARGET_RATIO)
    to_move = target_test - current_test

    if to_move > 0:
        print(f"🔄 Moving {to_move} '{emotion}' images from train ➝ test")

        train_emotion_path = os.path.join(TRAIN_PATH, emotion)
        test_emotion_path = os.path.join(TEST_PATH, emotion)

        aug_images = get_augmented_images(train_emotion_path)
        if len(aug_images) < to_move:
            print(f"⚠️ Not enough images to move for {emotion}. Needed: {to_move}, Available: {len(aug_images)}")
            to_move = len(aug_images)

        selected = random.sample(aug_images, to_move)
        for img in selected:
            src = os.path.join(train_emotion_path, img)
            dst = os.path.join(test_emotion_path, img)
            shutil.move(src, dst)

        print(f"✅ Moved {len(selected)} images to testing set.")

# Step 3: Recalculate new training set image counts
new_training_counts = {}
for emotion in os.listdir(TRAIN_PATH):
    path = os.path.join(TRAIN_PATH, emotion)
    count = len([f for f in os.listdir(path) if f.endswith(".jpg")])
    new_training_counts[emotion] = count

# Step 4: Compute class weights for cost-sensitive learning
total_images = sum(new_training_counts.values())
num_classes = len(new_training_counts)
weights = [
    total_images / (num_classes * new_training_counts[emotion])
    for emotion in sorted(new_training_counts)
]
weights_tensor = torch.tensor(weights, dtype=torch.float32)

# Step 5: Print stats
print("\n📊 Updated Training Set Counts:")
for cls, count in sorted(new_training_counts.items()):
    print(f"{cls}: {count}")

print("\n📐 Class Weights Tensor (for CrossEntropyLoss):")
print(weights_tensor)

# Step 6: Recalculate updated testing set counts
updated_testing_counts = {}
for emotion in os.listdir(TEST_PATH):
    path = os.path.join(TEST_PATH, emotion)
    count = len([f for f in os.listdir(path) if f.endswith(".jpg")])
    updated_testing_counts[emotion] = count



import os
import shutil
import random
import glob

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
EXTERNAL_DATA_PATH = "C:\\Users\\PC\\CBS_FYP2\\train_data_categorized"
TRAIN_PATH = "CASME2_augmented_with_apex_frames"
TEST_PATH = "CASME2_augmented_apex_frames_for_testing"
TARGET_RATIO = 0.2  # 20% testing

# ─────────────────────────────────────────────────────
# Step 1: Add new external fear and sadness images
# ─────────────────────────────────────────────────────
external_map = {
    "fear": os.path.join(EXTERNAL_DATA_PATH, "fear"),
    "sadness": os.path.join(EXTERNAL_DATA_PATH, "sadness"),
}

print("\n📥 Adding external data to training set for 'fear' and 'sadness'...")
for emotion, ext_path in external_map.items():
    if not os.path.exists(ext_path):
        print(f"❌ External path not found for {emotion}: {ext_path}")
        continue

    train_emotion_path = os.path.join(TRAIN_PATH, emotion)
    os.makedirs(train_emotion_path, exist_ok=True)

    image_paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(ext_path, ext)))
    print(f"Found {len(image_paths)} images for {emotion}. Adding to training set...")

    for idx, img_path in enumerate(image_paths):
        filename = f"external_{idx}.jpg"
        dst_path = os.path.join(train_emotion_path, filename)
        shutil.copy2(img_path, dst_path)

    print(f"✅ Added {len(image_paths)} external {emotion} images")

# ─────────────────────────────────────────────────────
# Step 2: Recalculate counts
# ─────────────────────────────────────────────────────
new_training_counts = {
    emotion: len([f for f in os.listdir(os.path.join(TRAIN_PATH, emotion)) if f.endswith(".jpg")])
    for emotion in os.listdir(TRAIN_PATH)
}

new_testing_counts = {
    emotion: len([f for f in os.listdir(os.path.join(TEST_PATH, emotion)) if f.endswith(".jpg")])
    for emotion in os.listdir(TEST_PATH)
}

# ─────────────────────────────────────────────────────
# Step 3: Rebalance to 80/20 split
# ─────────────────────────────────────────────────────
print("\n🔁 Rebalancing to achieve ~80/20 split...")

for emotion in new_training_counts:
    train_count = new_training_counts[emotion]
    test_count = new_testing_counts.get(emotion, 0)
    total = train_count + test_count
    target_test = int(total * TARGET_RATIO)

    to_move = target_test - test_count
    if to_move <= 0:
        continue  # Already sufficient

    print(f"🔄 Moving {to_move} samples from train ➝ test for '{emotion}'")

    train_emotion_path = os.path.join(TRAIN_PATH, emotion)
    test_emotion_path = os.path.join(TEST_PATH, emotion)
    os.makedirs(test_emotion_path, exist_ok=True)

    candidates = [f for f in os.listdir(train_emotion_path) if f.endswith(".jpg")]
    random.shuffle(candidates)
    selected = candidates[:to_move]

    for fname in selected:
        shutil.move(os.path.join(train_emotion_path, fname), os.path.join(test_emotion_path, fname))

    print(f"✅ Rebalanced '{emotion}': moved {len(selected)} images")

# ─────────────────────────────────────────────────────
# Step 4: Final Stats
# ─────────────────────────────────────────────────────
final_train_counts = {
    emotion: len([f for f in os.listdir(os.path.join(TRAIN_PATH, emotion)) if f.endswith(".jpg")])
    for emotion in os.listdir(TRAIN_PATH)
}

final_test_counts = {
    emotion: len([f for f in os.listdir(os.path.join(TEST_PATH, emotion)) if f.endswith(".jpg")])
    for emotion in os.listdir(TEST_PATH)
}

print("\n📊 Final Image Counts (Train/Test):")
for emotion in sorted(final_train_counts.keys()):
    train = final_train_counts[emotion]
    test = final_test_counts.get(emotion, 0)
    ratio = test / train if train > 0 else 0
    print(f"{emotion}: Test {test} / Train {train} = {ratio:.2f}")

print("\n📊 Updated Testing Set Counts:")
for cls, count in sorted(final_test_counts.items()):
    print(f"{cls}: {count}")

print("\n📐 Test/Train Ratios (per class):")
for cls in sorted(final_train_counts.keys()):
    test_count = final_test_counts.get(cls, 0)
    train_count = final_train_counts.get(cls, 1)  # avoid div by zero
    ratio = test_count / train_count
    print(f"{cls}: {test_count} / {train_count} = {ratio:.2f}")

# ─────────────────────────────────────────────────────
# Step 5: Compute Class Weights (at the end of pipeline)
# ─────────────────────────────────────────────────────
import torch

total_images = sum(final_train_counts.values())
num_classes = len(final_train_counts)

weights = [
    total_images / (num_classes * final_train_counts[emotion])
    for emotion in sorted(final_train_counts)
]

weights_tensor = torch.tensor(weights, dtype=torch.float32)

# Print results
print("\n📦 Final Class Weights Tensor (for CrossEntropyLoss):")
for emotion, weight in zip(sorted(final_train_counts), weights):
    print(f"{emotion}: {weight:.4f}")

print("\n📐 PyTorch Tensor Format:")
print(weights_tensor)