import tensorflow as tf
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "NISTDB4_RAW"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "class1_arc",
    "class2_whorl",
    "class3_loop",
]


# -----------------------------
# Load dataset
# -----------------------------

def load_dataset(directory):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    # Normalize pixel values from [0, 255] to [0, 1]
    dataset = dataset.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# -----------------------------
# Load train / validation / test
# -----------------------------

train_dir = DATASET_DIR / "train_set"
val_dir = DATASET_DIR / "val_set"
test_dir = DATASET_DIR / "test_set"

train_dataset = load_dataset(train_dir)
val_dataset = load_dataset(val_dir)
test_dataset = load_dataset(test_dir)


# -----------------------------
# Display dataset information
# -----------------------------

print("\nDataset loaded successfully!")

print("\nClass mapping:")
for index, class_name in enumerate(CLASS_NAMES):
    print(f"{index} -> {class_name}")

for images, labels in train_dataset.take(1):
    print("\nTraining batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image minimum:", tf.reduce_min(images).numpy())
    print("Image maximum:", tf.reduce_max(images).numpy())