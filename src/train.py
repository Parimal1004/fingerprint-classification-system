import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import train_dataset, val_dataset
from model import build_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPOCHS = 20

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Check GPU
# --------------------------------------------------

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("\nWARNING: No GPU detected. Training will use CPU.")


# --------------------------------------------------
# Build model
# --------------------------------------------------

model = build_model()

print("\nModel created successfully.\n")
model.summary()



# --------------------------------------------------
# Callbacks
# --------------------------------------------------

checkpoint_path = MODEL_DIR / "best_fingerprint_model.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------------------------------
# Save final model
# --------------------------------------------------

final_model_path = MODEL_DIR / "fingerprint_model_final.keras"

model.save(final_model_path)

print(f"\nFinal model saved to: {final_model_path}")
print(f"Best model saved to: {checkpoint_path}")


# --------------------------------------------------
# Plot training history
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

accuracy_plot = RESULTS_DIR / "accuracy_curve.png"

plt.savefig(
    accuracy_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Plot loss
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

loss_plot = RESULTS_DIR / "loss_curve.png"

plt.savefig(
    loss_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(f"Accuracy curve saved to: {accuracy_plot}")
print(f"Loss curve saved to: {loss_plot}")

print("\nTraining completed!")