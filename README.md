# Object Detection with Traditional Algorithms and Feature Extraction

This repository contains the implementation and analysis of an object detection system using traditional machine learning algorithms and feature extraction techniques. The project focuses on identifying objects in images and locating them via bounding boxes.

## Project Overview

The objective of this project is to detect objects in images and determine their locations using a combination of traditional classification algorithms and sliding window methods. This approach avoids reliance on deep learning and emphasizes handcrafted features for object detection and localization.

### Key Components:
1. **Object Classification**: Identify whether objects exist in given image segments.
2. **Object Localization**: Identify and mark the locations of objects within the images using bounding boxes.

---

## Dataset and Input/Output

- **Input**: 
  - Images with fixed viewpoints (e.g., top-down or specific angles). 
  - Example datasets include parking lots (top-down view) and pipe openings.

- **Output**: 
  - Classification: Predicted class and probability of detected objects.
  - Localization: Bounding box coordinates of detected objects in the image.

---

## Features and Methods

### 1. **Feature Extraction**
- **Canny Edge Detection**:
  - Detects edges using gradients and thresholds.
  - Key steps: Gaussian blur, gradient computation, non-maximum suppression, double thresholding, and edge tracking.

- **Histogram of Oriented Gradients (HOG)**:
  - Extracts features based on gradient orientations.
  - Parameters include `pixels_per_cell` and `cells_per_block` for better accuracy.

### 2. **Classification Algorithms**
- **K-Nearest Neighbors (KNN)**:
  - Classifies based on proximity in feature space.
  - Achieved higher accuracy using correlation distance metrics.
- **Logistic Regression**:
  - Predicts classes using a sigmoid function for binary classification.
- **Support Vector Machine (SVM)**:
  - Utilizes hyperplanes for classification with support for both linear and kernel-based methods.

## Sliding Window Technique

### Overview
- Scans image with fixed-size window
- Detects objects at multiple locations/scales
- Classifies each sub-window for potential objects

### Process
1. Move window across image
2. Classify each window
3. Collect potential object locations

### Key Characteristics
- Multi-scale detection
- Simple implementation
- Flexible approach

## Non-Maximum Suppression (NMS)

### Purpose
Non-Maximum Suppression eliminates redundant and overlapping bounding boxes, keeping only the most confident detections.

### Algorithm Steps
1. **Sorting**: Rank bounding boxes by confidence score
2. **Selection**: 
   - Choose highest-confidence box
   - Compare with remaining boxes using Intersection over Union (IoU)
3. **Elimination**: Remove boxes with IoU above threshold

### IoU Calculation
- **Intersection**: Overlapping area between boxes
- **Union**: Total area of both boxes
- **IoU Formula**: IoU = Intersection / Union

## Performance Comparison

| Object Type | Feature | Algorithm | Accuracy | Precision | Recall | IoU |
|------------|---------|-----------|----------|-----------|--------|-----|
| Car | HOG 8x8 | KNN | 0.95 | 0.94 | 0.95 | 0.75 |
| Car | HOG 8x8 | Logistic Regression | 0.95 | 0.94 | 0.95 | - |
| Car | HOG 8x8 | SVM | 0.97 | 0.97 | 0.97 | 0.75 |
| Pipe | HOG 16x16 | KNN | 0.94 | 0.94 | 0.94 | 0.63 |
| Pipe | HOG 16x16 | Logistic Regression | 0.96 | 0.96 | 0.96 | - |
| Pipe | HOG 16x16 | SVM | 0.97 | 0.97 | 0.97 | 0.63 |
| Human | HOG 16x16 | KNN | 0.89 | 0.90 | 0.89 | 0.75 |
| Human | HOG 16x16 | Logistic Regression | 0.85 | 0.85 | 0.85 | - |
| Human | HOG 16x16 | SVM | 0.90 | 0.90 | 0.90 | 0.75 |

---

## Project Conclusions

### Key Findings
- HOG features superior to Canny Edge detection
- SVM showed highest accuracy across object types
- Effective for simple object detection (pipes, cars)

### Limitations
- Slow processing speed
- Ineffective for complex object shapes
- Independent classification and localization models

### Recommendations
- Optimize feature extraction
- Develop specialized methods for different object types
- Improve model training dataset
- Focus on simple object detection scenarios

### Future Work
- Enhanced feature extraction techniques
- Faster processing algorithms
- Expand to more complex object detection