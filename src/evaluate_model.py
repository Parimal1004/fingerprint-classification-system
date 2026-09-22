import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import test_dataset

# Load the best trained model
model = load_model("models/best_fingerprint_model.keras")

# Class names
class_names = ["Arch", "Whorl", "Loop"]

# Get predictions
y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# Plot confusion matrix
plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Fingerprint Classification Confusion Matrix")

plt.tight_layout()

output_path = "../results/confusion_matrix.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

plt.close()

print(f"\nConfusion matrix saved to: {output_path}")
