# Traffic Sign Classification using HOG, Random Forest, and SVM

## Overview

This project focuses on **classifying Vietnamese traffic signs** using image processing and machine learning techniques. It was developed as the final project for the course *Introduction to Computer Vision (CS231.P11)* at UIT – VNUHCM.

We implemented and compared two supervised learning models:
- **Random Forest**
- **Support Vector Machine (SVM)**

Both models are trained on features extracted using the **Histogram of Oriented Gradients (HOG)** technique.

---

## Problem Statement

**Goal**: Predict the correct traffic sign class given an input image and its bounding box region.

- **Input**: An image containing one or more traffic signs (with bounding box coordinates provided).
- **Output**: The class label of the detected traffic sign.

---

## Dataset

The dataset consists of **10,873 labeled images**, split into:
- **Training set**: 9,471 images
- **Validation set**: 785 images
- **Test set**: 617 images

### Traffic sign categories:
- **W - Warning**
- **I - Instruction**
- **R - Regulatory**
- **P - Prohibition**
- **S - Supplementary**

Some images contain multiple signs. The data was collected and preprocessed via Roboflow and public datasets such as *Street Traffic Signs in Vietnam - COCO*.

---

## Data Preprocessing

### Image Processing:
- Auto-orientation (remove EXIF)
- Resized to **640x640**
- Auto-contrast (contrast stretching)
- Adaptive thresholding

### Feature Extraction:
- **HOG (Histogram of Oriented Gradients)**
  - Orientations: 9
  - Pixels per cell: (8, 8)
  - Cells per block: (2, 2)
  - Block normalization: L2-Hys

---

## Models and Tuning

### 1. Random Forest
- Hyperparameter tuning using **GridSearchCV**
- Final parameters:
  - `n_estimators=150`
  - `max_depth=7`
  - `min_samples_leaf=1`
  - `min_samples_split=2`
  - `class_weight="balanced"`

### 2. Support Vector Machine (SVM)
- Model: `SVC` from `sklearn.svm`
- Kernel: RBF
- Tuned via **GridSearchCV** to optimize accuracy

---

## Evaluation Metrics

Models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

| Model           | Validation Accuracy | Test Accuracy |
|----------------|----------------------|----------------|
| Random Forest  | 0.89                 | 0.88           |
| SVM            | 0.98                 | 0.96           |

> SVM showed higher accuracy and better performance in most classes, especially for "Regulatory" signs.

---

## Confusion Matrix Insight

- Random Forest tends to misclassify *R* (Regulatory) as *P* (Prohibition)
- SVM performs better across all categories, especially when distinguishing visually similar signs.

---

## Future Improvements

- Integrate sign **detection** without relying on bounding box input.
- Collect more samples for rare classes (I, S).
- Apply **deep learning** models (e.g., CNNs) for end-to-end training.
- Enhance data augmentation techniques (color jitter, random occlusion).

---

## Authors

- **Phạm Đông Hưng** – 22520521  
- **Phan Công Minh** – 22520884  
- Course: *CS231.P11 – Introduction to Computer Vision*  
- Instructor: *Dr. Mai Tiến Dũng*  
- University of Information Technology – VNUHCM

---

## References

- Yao et al. "Traffic sign recognition using HOG-SVM and grid search"
- HOG explanation – phamdinhkhanh.github.io
- Random Forest & SVM – GeeksforGeeks
- OpenCV Docs: Adaptive Thresholding
- Dataset: *Street Traffic Signs in Vietnam – COCO*

---

