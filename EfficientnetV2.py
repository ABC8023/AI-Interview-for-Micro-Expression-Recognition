from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Recall

# Define image size
image_size = (128, 128)
batch_size = 32

# Define paths
TRAIN_DATASET_PATH = "C:\\Users\\PC\\CBS_FYP2\\CASME2_augmented_with_apex_frames"
TEST_DATASET_PATH = "C:\\Users\\PC\\CBS_FYP2\\CASME2_augmented_apex_frames_for_testing"

# Emotion labels (folder names)
emotions = ["disgust", "fear", "happiness", "repression", "sadness", "surprise"]
num_classes = len(emotions)

# # Data generators
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DATASET_PATH,
    target_size=image_size,     # Resize images to 64x64
    batch_size=batch_size,      # Number of images per batch
    class_mode='categorical'    # One-hot encoded labels
)

test_gen = test_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False               # Important for predictions and evaluation
)

# CNN Model
# Input shape must match your image size
input_tensor = Input(shape=(128, 12, 3))

# Load EfficientNetV2B0 without top layers
base_model = EfficientNetV2B0(input_tensor=input_tensor, include_top=False, weights='imagenet')

# Freeze all layers initially
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Optional: adjust number of frozen layers
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output_tensor = Dense(num_classes, activation='softmax')(x)

# Full model
model = Model(inputs=input_tensor, outputs=output_tensor)

loss = CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=loss,
    metrics=['accuracy', Recall(name='recall')]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Class weights (optional: if your training set is imbalanced)
class_weight_dict = {
    0: 0.6966,  # disgust
    1: 0.8759,  # fear
    2: 1.2004,  # happiness
    3: 1.2954,  # repression
    4: 0.7992,  # sadness
    5: 1.7651   # surprise
}

# Train
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=100,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# Save model
model.save("efficientnetv2_micro_expression_classifier.keras")

print("✅ EfficientNetV2 Model trained and saved successfully.")



# ==========================
# 🔥 Standalone Model Evaluation Script
# ==========================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    cohen_kappa_score
)

# ───────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────
MODEL_PATH = "efficientnetv2_micro_expression_classifier.keras"
TEST_DATASET_PATH = "C:\\Users\\PC\\CBS_FYP2\\CASME2_augmented_apex_frames_for_testing"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# ───────────────────────────────────────────────
# Load Model
# ───────────────────────────────────────────────
model = load_model(MODEL_PATH)

# ───────────────────────────────────────────────
# Prepare Test Data
# ───────────────────────────────────────────────
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ───────────────────────────────────────────────
# Evaluate
# ───────────────────────────────────────────────
test_loss, test_accuracy, test_recall = model.evaluate(test_gen, verbose=1)

print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"❌ Test Loss: {test_loss:.4f}")
print(f"📊 Test Recall: {test_recall:.4f}")

# ───────────────────────────────────────────────
# Predictions & Ground Truth
# ───────────────────────────────────────────────
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes
emotions = list(test_gen.class_indices.keys())
num_classes = len(emotions)

# 📊 Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=emotions))

# 📉 Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("\n📉 Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=emotions, yticklabels=emotions)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ⚖️ Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"\n⚖️ Balanced Accuracy: {balanced_acc * 100:.2f}%")

# Training vs Validation Accuracy & Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.legend()
plt.show()

# 📈 Per-Class Precision, Recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

df_metrics = pd.DataFrame({
    'Emotion': emotions,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

df_metrics.set_index('Emotion').plot(kind='bar', figsize=(10,5), ylim=(0,1))
plt.title('Per-Class Precision, Recall & F1 Score')
plt.ylabel('Score')
plt.xlabel('Emotion Class')
plt.legend(loc='lower right')
plt.xticks(rotation=45)
plt.show()

# 📈 ROC-AUC Score
y_test_bin = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
y_pred_bin = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes)
roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average="macro")
print(f"📈 ROC-AUC Score: {roc_auc:.2f}")

# 4️⃣ ROC Curve (One-vs-Rest)
plt.figure(figsize=(10,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(fpr, tpr, label=f"{emotions[i]} (AUC: {auc(fpr, tpr):.2f})")

plt.plot([0,1], [0,1], 'k--')  # Random baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Emotion')
plt.legend(loc='lower right')
plt.show()

# 🤝 Cohen’s Kappa Score
kappa = cohen_kappa_score(y_true, y_pred)
print(f"🤝 Cohen's Kappa Score: {kappa:.2f}")

print("✅ Model Performance Analysis Completed!")