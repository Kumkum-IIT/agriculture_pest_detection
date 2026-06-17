# Pest Detection using YOLOv8

## Overview

This project implements a complete object detection pipeline for identifying **24 rice pest species** using **YOLOv8**. The workflow includes:

* Dataset preprocessing and label generation
* Stratified train/validation split
* YOLO-format annotation creation
* Two-stage transfer learning strategy
* Data augmentation
* Model evaluation
* Test-time inference
* Competition submission generation

The solution is designed for the **Object Detection Week-10 DLP Competition** on Kaggle.

---

## Dataset

### Input Files

```text
train.csv
train/
test/
```

### Annotation Format

The training CSV contains:

```text
image_id
class_name
x_min
y_min
x_max
y_max
width
height
```

Bounding boxes are converted into YOLO format:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `[0,1]`.

---

## Classes

The dataset contains 24 pest categories:

1. RiceLeafRoller
2. RiceLeafCaterpillar
3. PaddyStemMaggot
4. AsiaticRiceBorer
5. YellowRiceBorer
6. RiceGallMidge
7. RiceStemfly
8. BrownPlantHopper
9. WhiteBackedPlantHopper
10. SmallBrownPlantHopper
11. RiceWaterWeevil
12. RiceLeafhopper
13. GrainSpreaderThrips
14. RiceShellPest
15. Grub
16. MoleCricket
17. Wireworm
18. WhiteMarginedMoth
19. BlackCutworm
20. LargeCutworm
21. YellowCutworm
22. RedSpider
23. CornBorer
24. Armyworm

---

## Data Preparation

### Label Generation

The notebook:

* Converts bounding boxes to YOLO format
* Clips invalid coordinates
* Removes malformed boxes
* Generates `.txt` label files for every image

Example label:

```text
0 0.451233 0.562144 0.187500 0.254321
```

---

### Train / Validation Split

A stratified split is performed using the **dominant class per image**.

```python
train_test_split(
    test_size=0.2,
    stratify=dominant_class,
    random_state=42
)
```

Result:

```text
80% Training Images
20% Validation Images
```

---

## Data Augmentation

### Albumentations Pipeline

```python
HorizontalFlip
RandomBrightnessContrast
HueSaturationValue
GaussNoise
CoarseDropout
Rotate
```

### YOLO Built-in Augmentations

```text
Mosaic
MixUp
HSV augmentation
Random scaling
Horizontal flipping
```

---

## Training Strategy

A two-stage transfer learning approach is used.

---

### Stage 1 — Frozen Backbone

Goal:

Train only the detection head while keeping pretrained backbone layers frozen.

Configuration:

```text
Model: YOLOv8m
Epochs: 25
Image Size: 768
Batch Size: 8
Freeze Layers: 10
Learning Rate: 0.01
```

Benefits:

* Faster convergence
* Stable adaptation to pest dataset
* Reduced overfitting in early training

---

### Stage 2 — Full Fine-Tuning

Goal:

Fine-tune the entire network after the detection head has adapted.

Configuration:

```text
Epochs: 75
Image Size: 768
Batch Size: 16
Freeze Layers: 0
Learning Rate: 0.001
Patience: 30
```

Benefits:

* End-to-end optimization
* Improved localization accuracy
* Higher mAP performance

---

## Hyperparameters

```yaml
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4

degrees: 5
translate: 0.1
scale: 0.5

fliplr: 0.5
mosaic: 1.0
mixup: 0.1

box: 7.5
cls: 0.5
dfl: 1.5
```

---

## Evaluation

Validation metrics are computed using:

```python
model.val()
```

Reported metrics:

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* Confusion Matrix
* Precision–Recall Curves

---

## Training Visualization

The notebook automatically generates:

```text
training_curves.png
```

Including:

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* Box Loss
* Classification Loss

A vertical marker indicates the transition between:

```text
Stage 1 → Stage 2
```

---

## Inference

Prediction settings:

```python
imgsz = 768
conf = 0.01
iou = 0.5
augment = True
```

Test-time augmentation (TTA) is enabled:

```python
augment=True
```

to improve detection robustness.

---

## Submission Format

Output file:

```text
submission.csv
```

Structure:

```csv
ImageID,PredictionString
```

Example:

```csv
img_001.jpg,"RiceLeafRoller 0.93 120 80 300 260"
```

Each prediction contains:

```text
ClassName Confidence x1 y1 x2 y2
```

Multiple detections are concatenated into a single string.

---

## Saved Artifacts

### Model Weights

```text
best_stage1.pt
best_stage2.pt
```

### Configuration Files

```text
data.yaml
hyp.yaml
classes.txt
```

### Evaluation Outputs

```text
confusion_matrix.png
PR_curve.png
results.csv
training_curves.png
```

---

## Requirements

Install dependencies:

```bash
pip install ultralytics albumentations
```

Main libraries:

* Python 3.12+
* PyTorch
* Ultralytics YOLOv8
* Albumentations
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Hardware

Training configuration used:

```text
GPU: NVIDIA Tesla T4
Framework: PyTorch + Ultralytics
Platform: Kaggle Notebooks
```

---

## Future Improvements

* YOLOv8x experimentation
* Weighted sampling for minority classes
* K-Fold cross validation
* Ensemble of multiple YOLO checkpoints
* Soft-NMS / Weighted Box Fusion
* Class-specific confidence thresholds

---

## Author

**Kumkum Kwatra**

Object Detection Week-10 DLP Competition Submission

Built using YOLOv8 Transfer Learning and Two-Stage Fine-Tuning.
