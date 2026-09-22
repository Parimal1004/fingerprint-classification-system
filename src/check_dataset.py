from pathlib import Path
from PIL import Image

DATASET_DIR = Path("data/NISTDB4_RAW")

classes = {
    "class1_arc": "Arch",
    "class2_whorl": "Whorl",
    "class3_loop": "Loop",
}

for folder, class_name in classes.items():
    folder_path = DATASET_DIR / "train_set" / folder

    images = list(folder_path.glob("*"))

    print(f"\n{class_name}")
    print(f"Number of files: {len(images)}")

    if images:
        image_path = images[0]

        try:
            with Image.open(image_path) as img:
                print(f"Example: {image_path.name}")
                print(f"Format: {img.format}")
                print(f"Size: {img.size}")
                print(f"Mode: {img.mode}")

        except Exception as e:
            print(f"Could not open image: {e}")