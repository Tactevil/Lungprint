# Lungprint
# Explainable CNN for Pediatric Pneumonia Classification

A research project for binary classification of pediatric chest X-rays (NORMAL vs. PNEUMONIA) using convolutional neural networks, with a focus on class imbalance, model comparison, and interpretability.

> **Note:** This is a research and educational project. It is **not** a clinical diagnostic tool and should not be used for medical decision-making.

---

## Overview

Pneumonia is a leading cause of illness in children under five. Chest X-ray interpretation requires expert radiologists and is often limited in low-resource settings. This project develops and evaluates CNN models to automatically classify pediatric chest X-rays as NORMAL or PNEUMONIA using the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney.

The study compares a custom CNN with transfer learning models (DenseNet121, ResNet50, EfficientNetB0), systematically evaluates class imbalance techniques, and applies Grad-CAM for explainability.

---

## Features

- Binary classification of chest X-rays (NORMAL / PNEUMONIA)
- Custom CNN and three pretrained transfer learning models
- Systematic comparison of class imbalance methods:
  - Class weights
  - Oversampling
  - Focal loss
  - Augmentation for minority class
- Evaluation using accuracy, precision, recall, F1-score, AUC-ROC, PR-AUC, and confusion matrix
- Grad-CAM heatmaps for model interpretability
- Reproducible pipeline with fixed random seeds
- Test set kept untouched until final evaluation

---

## Dataset

**Name:** Chest X-Ray Images (Pneumonia)  
**Creator:** Paul Mooney  
**Source:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
**Total Images:** 5,856  
**Classes:** NORMAL, PNEUMONIA  
**Split:** Pre-divided into training, validation, and test sets  
**Population:** Pediatric patients aged 1–5 years  

The dataset is class imbalanced (more pneumonia than normal images). The test set is reserved for final evaluation only.

---

## Models

| Model | Description |
|-------|-------------|
| Baseline CNN | 3–4 convolutional blocks with pooling, dropout, and dense layers |
| DenseNet121 | Pretrained on ImageNet, fine-tuned |
| ResNet50 | Pretrained on ImageNet, fine-tuned |
| EfficientNetB0 | Pretrained on ImageNet, fine-tuned |

---

## Project Structure

Lungprints/
├── data/ # Dataset
│ ├── train/
│ ├── val/
│ └── test/
├── src/
│ ├── data_loader.py # Preprocessing and augmentation
│ ├── models.py # Model definitions
│ ├── train.py # Training loop
│ ├── evaluate.py # Evaluation metrics
│ ├── imbalance.py # Imbalance handling techniques
│ └── gradcam.py # Grad-CAM visualization
├── notebooks/ # Exploratory analysis and experiments
├── results/ # Saved models, logs, plots
├── requirements.txt
└── README.md
