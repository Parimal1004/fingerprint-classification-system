"""
Grad-CAM style explanations for the fingerprint CNN.

The model ends in Flatten -> Dense, so the gradient differs from one image
location to the next. Classic Grad-CAM averages that gradient over space,
which throws away exactly the information we want (and can cancel out to
an empty map). This module therefore uses the element-wise version
(gradient x activation, as in HiResCAM) by default.

Used by app.py, and can also be run directly on one image:

    python gradcam.py path/to/fingerprint.png
"""

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_fingerprint_model.keras"
RESULTS_DIR = PROJECT_DIR / "results"

CLASS_NAMES = ["Arch", "Whorl", "Loop"]
IMAGE_SIZE = (224, 224)

_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


# --------------------------------------------------
# Computation (needs TensorFlow)
# --------------------------------------------------

def find_last_conv_layer(model):
    """Return the last Conv2D layer of the model."""
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    raise ValueError("No Conv2D layer found in the model.")


def compute_all_cams(model, image_array, method="hirescam"):
    """
    Raw (un-normalised, positive-only) evidence maps for every class.

    method:
      "hirescam" - gradient x activation at every location (default)
      "gradcam"  - classic Grad-CAM (gradient averaged over space)

    Returns (cams, probabilities) where cams has shape (n_classes, h, w).
    """
    import tensorflow as tf

    conv_layer = find_last_conv_layer(model)
    final_layer = model.layers[-1]

    x = tf.convert_to_tensor(image_array, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        conv_output = None

        # Forward pass up to (not including) the final softmax layer.
        # training=False keeps augmentation and dropout switched off.
        for layer in model.layers[:-1]:
            x = layer(x, training=False)

            if layer is conv_layer:
                conv_output = x
                tape.watch(conv_output)

        # Work with raw scores (logits). Softmax saturates on confident
        # predictions, which makes its gradients vanish.
        if isinstance(final_layer, tf.keras.layers.Dense):
            kernel, bias = final_layer.get_weights()
            scores = tf.matmul(x, kernel) + bias
            probabilities = tf.nn.softmax(scores)[0]
        else:
            raise ValueError("Expected the last layer to be a Dense layer.")

        n_classes = int(scores.shape[-1])

        # Target for class c = log-odds of c against all other classes:
        #     score_c - logsumexp(scores of the other classes)
        # This accounts for competition between classes, so "Arch" is
        # explained as "Arch rather than Whorl/Loop".
        targets = []
        for c in range(n_classes):
            others = tf.concat([scores[:, :c], scores[:, c + 1:]], axis=1)
            targets.append(scores[:, c] - tf.reduce_logsumexp(others, axis=1))

    cams = []
    for target in targets:
        grads = tape.gradient(target, conv_output)

        if method == "gradcam":
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(conv_output[0] * weights, axis=-1)
        else:
            cam = tf.reduce_sum(conv_output[0] * grads[0], axis=-1)

        cams.append(tf.nn.relu(cam).numpy())

    del tape

    return np.stack(cams).astype(np.float32), probabilities.numpy()


def compute_gradcam(model, image_array, class_index=None, method="hirescam"):
    """
    Heatmap for one class.

    Parameters
    ----------
    model : the loaded Keras model
    image_array : float32 array, shape (1, H, W, 1), values in [0, 1]
    class_index : class to explain; None means the predicted class

    Returns
    -------
    heatmap : float32 array (h, w) in [0, 1]
    probabilities : numpy array of class probabilities
    class_index : the class that was explained

    All classes share one colour scale (the strongest evidence for any
    class is 1.0), so a class the model did not choose shows up faint
    instead of being stretched to full red.
    """
    cams, probabilities = compute_all_cams(model, image_array, method)

    if class_index is None:
        class_index = int(np.argmax(probabilities))

    reference = cams.max()
    heatmap = cams[class_index] / reference if reference > 0 else cams[class_index]

    return heatmap.astype(np.float32), probabilities, class_index


# --------------------------------------------------
# Visualisation helpers (no TensorFlow needed)
# --------------------------------------------------

def resize_heatmap(heatmap, size):
    """Upscale a small heatmap to size=(width, height), values stay in [0, 1]."""
    heatmap = np.clip(heatmap, 0.0, 1.0)
    image = Image.fromarray(np.uint8(heatmap * 255))
    image = image.resize(size, _BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def colorize_heatmap(heatmap, size=IMAGE_SIZE):
    """Return the heatmap as a colour (jet) PIL image of the given size."""
    resized = resize_heatmap(heatmap, size)
    rgb = colormaps["jet"](resized)[..., :3]
    return Image.fromarray(np.uint8(rgb * 255))


def overlay_heatmap(gray_image, heatmap, alpha=0.5):
    """
    Blend the heatmap onto the fingerprint.

    The blend strength follows the heatmap itself, so low-importance areas
    stay as the original image and ridges remain visible.
    """
    base = gray_image.convert("L").convert("RGB")
    size = base.size

    base_array = np.asarray(base, dtype=np.float32)
    color_array = np.asarray(colorize_heatmap(heatmap, size), dtype=np.float32)

    strength = (alpha * resize_heatmap(heatmap, size))[..., np.newaxis]
    blended = base_array * (1.0 - strength) + color_array * strength

    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


# --------------------------------------------------
# Command line use
# --------------------------------------------------

if __name__ == "__main__":
    import sys

    from tensorflow.keras.models import load_model

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python gradcam.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    model = load_model(MODEL_PATH)

    gray = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    array = array[np.newaxis, ..., np.newaxis]

    print(f"\nImage: {image_path.name}")

    panels = [("Input", gray.convert("RGB"))]

    for method in ["hirescam", "gradcam"]:
        cams, probabilities = compute_all_cams(model, array, method)
        reference = cams.max()

        print(f"\nMethod: {method}")
        print(f"Predicted: {CLASS_NAMES[int(np.argmax(probabilities))]}")

        for i, name in enumerate(CLASS_NAMES):
            peak = cams[i].max() / reference if reference > 0 else 0.0
            covered = (cams[i] > 0.5 * reference).mean() * 100 if reference > 0 else 0.0
            print(
                f"  {name:6s} prob {probabilities[i] * 100:6.2f}%   "
                f"relative peak {peak:.2f}   area above half-max {covered:5.1f}%"
            )

            if method == "hirescam":
                heat = cams[i] / reference if reference > 0 else cams[i]
                panels.append((name, overlay_heatmap(gray, heat, alpha=0.6)))

    width, height = IMAGE_SIZE
    combined = Image.new("RGB", (width * len(panels), height))
    draw = ImageDraw.Draw(combined)

    for i, (label, panel) in enumerate(panels):
        combined.paste(panel, (i * width, 0))
        draw.rectangle([i * width, 0, i * width + 90, 16], fill=(0, 0, 0))
        draw.text((i * width + 4, 2), label, fill=(255, 255, 255))

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"gradcam_{image_path.stem}.png"
    combined.save(output_path)

    print(f"\nSaved: {output_path}")