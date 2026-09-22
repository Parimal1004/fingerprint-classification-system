import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


MODEL_PATH = "../models/best_fingerprint_model.keras"

CLASS_NAMES = ["Arch", "Whorl", "Loop"]

IMAGE_SIZE = (224, 224)


def predict_fingerprint(image_path):
    # Load model
    model = load_model(MODEL_PATH)

    # Load image
    image = load_img(
        image_path,
        color_mode="grayscale",
        target_size=IMAGE_SIZE
    )

    # Convert image to NumPy array
    image = img_to_array(image)

    # Normalize pixel values
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image, verbose=0)[0]

    # Get predicted class
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[predicted_index] * 100

    print("\n" + "=" * 50)
    print("FINGERPRINT CLASSIFICATION")
    print("=" * 50)

    print(f"\nPredicted Class : {predicted_class}")
    print(f"Confidence      : {confidence:.2f}%")

    print("\nClass Probabilities:")

    for class_name, probability in zip(CLASS_NAMES, predictions):
        print(f"{class_name:10s}: {probability * 100:.2f}%")

    print("=" * 50)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    predict_fingerprint(image_path)
