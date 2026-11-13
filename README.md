# 🌾 Paddy Disease Classification & Severity Estimation

A Deep Learning-based Paddy Disease Classification system using **EfficientNet-B0**, **Grad-CAM**, and **K-Means Clustering** to classify paddy leaf diseases, visualize disease-related regions, and estimate disease severity.

## 📌 Project Overview

This project develops an automated paddy leaf disease classification pipeline using Deep Learning and Computer Vision.

The system uses an EfficientNet-B0 model with a custom classification head to identify diseases from paddy leaf images. Grad-CAM provides visual explanations of the model's predictions, while K-Means clustering (K=2) segments the Grad-CAM heatmap to estimate the affected region.

## 🎯 Key Features

- **Paddy Disease Classification:** Classifies paddy leaf images using EfficientNet-B0.
- **Transfer Learning:** Uses ImageNet-pretrained EfficientNet-B0 weights when available.
- **Data Augmentation:** Applies horizontal and vertical flipping, rotation, color jitter, and random erasing.
- **Grad-CAM Visualization:** Highlights image regions influencing the model's prediction.
- **Disease Severity Estimation:** Uses K-Means clustering to generate a binary mask from Grad-CAM activation maps.
- **Confidence Score:** Provides prediction confidence using Softmax probabilities.
- **Batch Inference:** Evaluates multiple test images and visualizes their predictions.
- **Model Checkpointing:** Saves the best model and supports training recovery.
- **Performance Evaluation:** Generates classification reports, confusion matrices, ROC curves, and Precision-Recall curves.

## 🧠 Model Architecture

The project follows this pipeline:

**Input Paddy Leaf Image → Image Preprocessing → EfficientNet-B0 → Classification Head → Disease Prediction**

For explainability and severity estimation:

**Grad-CAM → Activation Heatmap → K-Means Clustering (K=2) → Binary Infection Mask → Severity Estimation**

### Model Components

| Component | Purpose |
|---|---|
| EfficientNet-B0 | Feature extraction and classification |
| Custom Classification Head | Disease class prediction |
| Softmax | Prediction confidence |
| Grad-CAM | Visual explanation |
| K-Means (K=2) | Heatmap segmentation |
| Severity Estimation | Calculates affected-region percentage |

## 📊 Severity Estimation

The system estimates severity by segmenting the Grad-CAM heatmap into two clusters:

- Healthy / lower-activation region
- Infected / higher-activation region

The severity percentage is calculated from the proportion of pixels assigned to the higher-activation cluster.

### Severity Categories

| Severity Percentage | Category |
|---|---|
| Below 25% | Mild |
| 25% – Below 50% | Moderate |
| 50% – Below 75% | Severe |
| 75% – 100% | Very Severe |

The final severity score is confidence-weighted:

`Weighted Severity = Raw Severity × Prediction Confidence`

**Note:** This is a Grad-CAM-based heuristic estimation, not a clinically or agriculturally validated measurement of actual infected leaf area.

## 📂 Dataset

**Dataset:** Paddy Disease Classification

**Source:** Kaggle Competition

https://www.kaggle.com/competitions/paddy-disease-classification

The notebook automatically downloads the competition dataset using `kagglehub`.

The dataset is organized into class-specific image folders. The notebook discovers the available classes automatically.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL (Pillow)
- KaggleHub
- Google Colab / Kaggle Notebooks

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python numpy pandas
pip install matplotlib seaborn scikit-learn pillow kagglehub
```

### 3. Configure Kaggle API

To download the dataset outside Kaggle Notebooks:

1. Create a Kaggle account.
2. Go to Kaggle Settings.
3. Generate a Kaggle API token.
4. Configure the credentials securely using the `kaggle.json` file or environment variables.

Never upload your Kaggle API credentials to GitHub.

### 4. Run the Notebook

Open `paddy.ipynb` in Google Colab or Kaggle Notebooks.

Run the notebook cells sequentially:

1. Kaggle Authentication & Dataset Download
2. Install / Import Dependencies
3. Configuration & Data Directory Resolution
4. Dataset Discovery & Train/Validation/Test Split
5. Image Transformations & Dataset Class
6. DataLoaders
7. EfficientNet-B0 Model Initialization
8. Training & Evaluation
9. Checkpoint Recovery
10. Training Curves
11. Test Set Evaluation
12. Grad-CAM Implementation
13. K-Means Severity Estimation
14. End-to-End Inference
15. Sample Image Prediction
16. Result Visualization
17. Batch Inference
18. Custom Image Inference

## 🔬 Training Configuration

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Input Image Size | 224 × 224 |
| Batch Size | 64 |
| Maximum Epochs | 15 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | Cross-Entropy Loss |
| Label Smoothing | 0.1 |
| Train Split | 80% |
| Validation Split | 10% |
| Test Split | 10% |
| Early Stopping Patience | 4 |
| Random Seed | 42 |

The notebook uses mixed-precision training when CUDA is available to improve GPU memory efficiency and training speed.

## 📈 Evaluation & Visualization

The notebook generates:

- Training and validation loss curves
- Training and validation accuracy curves
- Classification report
- Balanced accuracy
- Confusion matrix (raw and normalized)
- Multiclass ROC curves and AUC
- Precision-Recall curves
- Grad-CAM heatmap overlays
- K-Means infection masks
- Batch prediction visualizations

## 💾 Model Checkpoint

The best-performing model checkpoint is saved as:

```text
best_paddy_model.pth
```

A separate resume checkpoint is also maintained to support recovery of interrupted training sessions.

## 📷 Sample Outputs

The notebook generates the following visualization files:

- `training_curves.png`
- `confusion_matrix.png`
- `gradcam_result.png`
- `batch_inference.png`
- `custom_inference.png`

These visualizations help inspect classification performance, model attention, and estimated severity.

## 🚀 Future Improvements

- Improve severity estimation using validated pixel-level infection annotations.
- Explore more advanced segmentation techniques.
- Evaluate additional CNN and Transformer architectures.
- Develop a web-based application for real-time disease prediction.
- Validate severity estimates against expert agricultural assessments.

## 👩‍💻 Author

Bijoy Sen

GitHub: https://github.com/bijoy1295

---

⭐ If you find this project useful, consider giving the repository a star!
