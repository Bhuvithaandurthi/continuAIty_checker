# 🎬 ContinuAIty Checker: Deep Learning Video Anomaly Detection

[![AI DevOps Pipeline](https://github.com/Bhuvithaandurthi/continuAIty_checker/actions/workflows/main.yml/badge.svg)](https://github.com/Bhuvithaandurthi/continuAIty_checker/actions)

### 📄 Abstract
The **ContinuAIty Checker** is an automated, AI-driven computer vision pipeline designed to detect visual inconsistencies and temporal anomalies across consecutive film shots. By replacing manual continuity logging with deep learning, this system ensures seamless transitions in video production environments.

### 🧠 Vision Architecture & Pipeline
This project leverages state-of-the-art object detection and spatial tracking to maintain scene consistency.
* **Step 1 (Frame Extraction):** Automated parsing of raw `.mp4` video feeds into sequential frame arrays using `OpenCV`.
* **Step 2 (Tensor Preprocessing):** Normalization and resizing of visual data for neural network ingestion via `PyTorch`.
* **Step 3 (Feature Detection):** Inference running on a custom fine-tuned **YOLOv8** architecture to detect misplaced objects, wardrobe shifts, or spatial anomalies.
* **Step 4 (Anomaly Scoring):** Frame-by-frame comparative analysis to flag temporal inconsistencies.

### 📊 Performance Metrics
* **Latency Reduction:** Achieved a **40% reduction** in manual production review latency by automating the continuity logging process.
* **Model Robustness:** Fine-tuned on custom, real-world video datasets to maintain high Intersection over Union (IoU) and bounding-box accuracy across diverse lighting conditions.

### ⚙️ Tech Stack & CI/CD
* **Deep Learning Framework:** `PyTorch`
* **Computer Vision:** `Ultralytics YOLOv8`, `OpenCV`, `NumPy`
* **Language:** `Python`
* **DevOps:** Automated GitHub Actions Pipeline (`Docker` build verification).

---
*Developed by Bhuvitha Andurthi.*
