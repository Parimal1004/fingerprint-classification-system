# 🔐 Fingerprint Classification System

A deep learning-based **fingerprint pattern classification system** that classifies fingerprint images into three pattern categories:

* 🏛️ **Arch**
* 🌀 **Whorl**
* 🔄 **Loop**

The system uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and provides an interactive Streamlit web interface for uploading fingerprint images and obtaining predictions with confidence scores.

> **Note:** This project performs **fingerprint pattern classification**, not fingerprint identity recognition. It determines whether a fingerprint belongs to the Arch, Whorl, or Loop pattern category.

---

## 📌 Project Overview

Fingerprint patterns are commonly categorized into three major classes: **Arch, Whorl, and Loop**. Manually identifying these patterns can be time-consuming and subjective.

This project applies deep learning to automatically classify fingerprint images into these three categories.

### Workflow

```text
Fingerprint Image
       ↓
Image Preprocessing
       ↓
Image Resizing (224 × 224)
       ↓
Normalization
       ↓
Data Augmentation
       ↓
Convolutional Neural Network
       ↓
Softmax Classification
       ↓
Arch / Whorl / Loop
       ↓
Prediction + Confidence Score
```

---

## 🎯 Objectives

* Build a CNN-based fingerprint pattern classifier.
* Classify fingerprint images into Arch, Whorl, and Loop.
* Apply image preprocessing and augmentation techniques.
* Train and evaluate the model using separate training, validation, and testing datasets.
* Visualize model performance using accuracy/loss curves and a confusion matrix.
* Develop an interactive Streamlit application for real-time predictions.

---

## 🗂️ Dataset

The project uses a fingerprint dataset based on **NIST Special Database 4 (DB4)**.

The dataset is organized into three classes:

```text
NISTDB4_RAW/
├── train_set/
│   ├── class1_Arc/
│   ├── class2_Whorl/
│   └── class3_Loop/
│
├── val_set/
│   ├── class1_Arc/
│   ├── class2_Whorl/
│   └── class3_Loop/
│
└── test_set/
    ├── class1_Arc/
    ├── class2_Whorl/
    └── class3_Loop/
```

### Dataset Distribution

| Dataset    |    Arch |   Whorl |    Loop |     Total |
| ---------- | ------: | ------: | ------: | --------: |
| Training   |     560 |     559 |     560 |     1,679 |
| Validation |     160 |     159 |     160 |       479 |
| Testing    |      81 |      81 |      81 |       243 |
| **Total**  | **801** | **799** | **801** | **2,401** |

Images are grayscale and resized to:

```text
224 × 224 × 1
```

### Dataset Privacy / Repository Size

The fingerprint dataset is **not included in this GitHub repository** because of its size and dataset distribution considerations.

The dataset directory is excluded through `.gitignore`.

---

## 🧠 Model Architecture

The classifier is a custom Convolutional Neural Network implemented using TensorFlow/Keras.

### Architecture

```text
Input
224 × 224 × 1
      ↓
Data Augmentation
      ↓
Conv2D (32 filters)
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D (64 filters)
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D (128 filters)
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D (256 filters)
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Flatten
      ↓
Dense (128)
      ↓
Dropout (0.5)
      ↓
Dense (3)
      ↓
Softmax
      ↓
Arch / Whorl / Loop
```

### Training Configuration

| Parameter             | Value                           |
| --------------------- | ------------------------------- |
| Framework             | TensorFlow / Keras              |
| Input Size            | 224 × 224 × 1                   |
| Number of Classes     | 3                               |
| Optimizer             | Adam                            |
| Initial Learning Rate | 0.001                           |
| Loss Function         | Sparse Categorical Crossentropy |
| Batch Size            | 32                              |
| Maximum Epochs        | 20                              |
| Dropout               | 0.5                             |
| Output Activation     | Softmax                         |
| Parameters            | ~6.81 million                   |

### Data Augmentation

The model applies light augmentation during training:

* Random Rotation
* Random Zoom
* Random Translation

These transformations help the model generalize to small variations in fingerprint image orientation and positioning.

---

## 📊 Model Performance

The trained model achieved:

### Test Accuracy

**87.65%**

### Test Loss

**0.4918**

### Classification Report

| Class                | Precision | Recall |   F1-Score | Support |
| -------------------- | --------: | -----: | ---------: | ------: |
| Arch                 |    92.31% | 88.89% |     90.57% |      81 |
| Whorl                |    85.39% | 93.83% |     89.41% |      81 |
| Loop                 |    85.53% | 80.25% |     82.80% |      81 |
| **Overall Accuracy** |           |        | **87.65%** | **243** |

### Confusion Matrix

```text
                Predicted
              Arch  Whorl  Loop

Actual Arch     72     3     6
Actual Whorl     0    76     5
Actual Loop      6    10    65
```

The confusion matrix and training curves are available in the `results/` directory.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit web application.

The application allows users to:

1. Upload a fingerprint image.
2. Preview the uploaded image.
3. Run the trained CNN model.
4. View the predicted fingerprint pattern.
5. View the prediction confidence.
6. See class-wise probability scores.
7. View model performance information.
8. View the confusion matrix.
9. Review the model methodology and architecture.

### Example Prediction

For a test fingerprint image, the model produced:

```text
Predicted Class: Whorl
Confidence: 99.30%
```

---

## 📁 Project Structure

```text
fingerprint-classification-system/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── NISTDB4_RAW/
│       └── # Dataset (not included in GitHub)
│
├── models/
│   └── best_fingerprint_model.keras
│
├── results/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── dataset_samples.png
│   └── loss_curve.png
│
├── src/
│   ├── app.py
│   ├── check_dataset.py
│   ├── data_loader.py
│   ├── evaluate_model.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── visualize_dataset.py
│
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Image Processing

* OpenCV
* Pillow

### Data Science

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

### Web Application

* Streamlit

### Development Environment

* Ubuntu on WSL2
* NVIDIA CUDA
* NVIDIA GeForce RTX 3050 Laptop GPU

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Parimal1004/fingerprint-classification-system.git
cd fingerprint-classification-system
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

Install the required Python packages:

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib seaborn streamlit pillow
```

> GPU-specific TensorFlow/CUDA dependencies may require additional configuration depending on the operating system and hardware.

---

## 📂 Dataset Setup

Place the dataset inside:

```text
data/NISTDB4_RAW/
```

The expected structure is:

```text
data/NISTDB4_RAW/
├── train_set/
├── val_set/
└── test_set/
```

Each split should contain:

```text
class1_Arc/
class2_Whorl/
class3_Loop/
```

You can verify the dataset using:

```bash
cd src
python check_dataset.py
```

---

## 🚀 Running the Project

### Run the Streamlit Application

From the `src` directory:

```bash
cd src
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## 🔮 Make a Prediction from the Command Line

A single fingerprint image can also be classified using:

```bash
cd src
python predict.py "<path_to_image>"
```

Example:

```bash
python predict.py "../data/NISTDB4_RAW/test_set/class2_Whorl/class2_Whorl_0717.png"
```

The output will look like:

```text
Predicted Class: Whorl
Confidence: 99.30%

Class Probabilities:
Arch: 0.06%
Whorl: 99.30%
Loop: 0.64%
```

---

## 🏋️ Training the Model

To train the CNN from scratch:

```bash
cd src
python train.py
```

The training process includes:

* Training/validation dataset loading
* Data augmentation
* CNN training
* Model checkpointing
* Early stopping
* Learning-rate reduction
* Accuracy/loss visualization
* Final model saving

The best model is saved as:

```text
models/best_fingerprint_model.keras
```

---

## 📈 Evaluation

To evaluate the trained model:

```bash
cd src
python evaluate_model.py
```

The evaluation generates classification metrics and a confusion matrix for the test dataset.

---

## 📊 Results and Visualizations

The `results/` directory contains:

### Accuracy Curve

Shows the training and validation accuracy throughout training.

### Loss Curve

Shows the training and validation loss throughout training.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each fingerprint pattern.

### Dataset Samples

Provides a visual sample of images from the different fingerprint classes.

---

## 🔬 Methodology

The system follows these major stages:

### 1. Data Collection

Fingerprint images are organized into Arch, Whorl, and Loop categories.

### 2. Preprocessing

Images are:

* Converted to grayscale
* Resized to 224 × 224
* Normalized to the range `[0, 1]`

### 3. Data Augmentation

Small transformations are applied to training images to improve generalization.

### 4. CNN Feature Extraction

Multiple convolutional layers automatically learn visual features from fingerprint patterns.

### 5. Classification

The extracted features are passed through fully connected layers and a Softmax output layer.

### 6. Evaluation

The trained model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### 7. Deployment

The trained model is integrated into a Streamlit application for interactive predictions.

---

## ⚠️ Limitations

* The model classifies only three fingerprint pattern categories: Arch, Whorl, and Loop.
* It does not identify or authenticate individuals.
* Performance may vary on fingerprint images with significant noise, cropping, rotation, or image-quality differences from the training dataset.
* The current dataset is relatively small compared with large-scale deep learning datasets.
* The trained model may require additional optimization for deployment on resource-constrained devices.

---

## 🔮 Future Improvements

Possible future enhancements include:

* Increasing the size and diversity of the training dataset.
* Applying advanced fingerprint preprocessing techniques.
* Experimenting with transfer learning architectures such as MobileNet, EfficientNet, or ResNet.
* Improving Loop classification performance.
* Adding automatic image quality assessment.
* Adding fingerprint orientation/ridge-flow analysis.
* Deploying the Streamlit application online.
* Optimizing the model for mobile or edge devices.
* Comparing multiple CNN architectures.
* Adding explainability techniques such as Grad-CAM to visualize regions influencing predictions.

---

## 👨‍💻 Author

**Parimal Goud**

AI & Data Science Undergraduate

GitHub: [@Parimal1004](https://github.com/Parimal1004)

---

## 📜 Disclaimer

This project is developed for **educational and research purposes** and demonstrates the use of deep learning for fingerprint pattern classification.

It is not intended to be used as a biometric identification, authentication, forensic, or security system without appropriate validation, testing, and safeguards.
