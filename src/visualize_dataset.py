from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


DATASET_DIR = Path("data/NISTDB4_RAW")

classes = {
    "class1_arc": "Arch",
    "class2_whorl": "Whorl",
    "class3_loop": "Loop",
}

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for row, (folder, class_name) in enumerate(classes.items()):
    folder_path = DATASET_DIR / "train_set" / folder
    image_paths = list(folder_path.glob("*.png"))

    selected_images = random.sample(image_paths, 3)

    for col, image_path in enumerate(selected_images):
        with Image.open(image_path) as img:
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(class_name)
            axes[row, col].axis("off")

plt.suptitle("Fingerprint Dataset Samples", fontsize=16)
plt.tight_layout()

output_path = Path("results/dataset_samples.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved visualization to: {output_path}")