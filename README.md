# Pneumonia Detection from Chest X-rays
A deep learning system that detects pneumonia from chest X-ray images using transfer learning and explainable AI. Built with PyTorch and deployable via FastAPI.

Overview
This project trains a high-accuracy binary classifier to distinguish between normal and pneumonia-affected chest X-rays. It uses the Kaggle Chest X-ray Pneumonia dataset, enhanced with additional data to improve generalization. The model leverages EfficientNet-B0 with transfer learning and includes Grad-CAM for visual interpretability, making it suitable for medical decision support.

Key Features
Trained on pediatric chest X-rays with binary labels: NORMAL and PNEUMONIA
Achieves AUC > 0.92 and 100% sensitivity (critical for avoiding missed pneumonia cases)
Uses weighted loss to handle class imbalance
Includes Grad-CAM visualization to show which lung regions drive predictions
Deployed via FastAPI for easy integration
Full training pipeline in Google Colab (beginner-friendly, GPU-supported)
Model Performance
Test Accuracy: ~85%
AUC: ~0.92
Sensitivity (Recall for Pneumonia): 86%
Specificity: ~85%
Precision (for Pneumonia): 0.8125
The model prioritizes high sensitivity — it aims to catch every possible pneumonia case, even at the cost of some false positives. This aligns with clinical safety standards.

Dataset
Primary source: Chest X-ray Pneumonia (Kaggle)
Augmented with additional data to reduce overfitting and data leakage risks
Images are preprocessed: resized to 224×224, normalized using ImageNet stats, and augmented during training (flips, rotation, color jitter)

Architecture
Backbone: EfficientNet-B0 (pre-trained on ImageNet)
Classifier Head: Linear layer with dropout
Loss Function: Weighted Cross-Entropy
Optimizer: AdamW with learning rate scheduling
Training: Mixed-precision, 20+ epochs, batch size 32
Deployment
The model is served via a FastAPI endpoint:

POST /predict: Accepts an image file, returns:
Prediction ("Normal" or "Pneumonia")
Confidence score (0.0–1.0)
Grad-CAM heatmap (optional, for interpretability)

To run locally:
pip install -r requirements.txt
uvicorn app:app --reload

Limitations
Trained primarily on pediatric X-rays; performance on adult images may vary
Dataset may contain duplicates (known issue with Kaggle source), mitigated by adding external data
Not a diagnostic tool — intended for research and educational use only

Disclaimer
This model is not approved for clinical use. Always consult a licensed radiologist for medical diagnosis.
