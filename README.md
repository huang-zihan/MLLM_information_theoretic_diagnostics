# Concept Bottleneck Models for Multimodal Understanding

## Overview

This repository implements a comprehensive **Concept Bottleneck Model (CBM)** pipeline for multimodal understanding tasks, particularly focusing on visual question answering (VQA). The framework extracts interpretable visual concepts from images and uses them as bottlenecks for downstream reasoning tasks. The system supports both training on large-scale datasets (COCO2014) and evaluation on multiple benchmarks including Pope, CHAIR, HAL, and VQA-v2.

<!-- ## Mathematical Framework

Given an image $I \in \mathbb{R}^{H \times W \times 3}$ and a textual question $Q$, our CBM framework operates through three stages:

### 1. Concept Extraction
Object-level concepts are extracted using a pretrained detector:
$$
C_{\text{obj}} = f_{\text{detector}}(I) = \{(c_i, b_i, s_i)\}_{i=1}^{N}
$$
where $c_i$ denotes concept class, $b_i$ the bounding box, and $s_i$ the confidence score.

### 2. Multimodal Feature Encoding
LLaVA processes both visual and textual inputs:
$$
F_{\text{LLaVA}} = \phi_{\text{LLaVA}}(I, Q) \in \mathbb{R}^{d}
$$

### 3. Concept Bottleneck Prediction
The bottleneck layer learns to predict concepts from features:
$$
P(C|I,Q) = \sigma(W \cdot F_{\text{LLaVA}} + b)
$$
where $\sigma$ is the sigmoid function, $W \in \mathbb{R}^{d \times K}$, $b \in \mathbb{R}^{K}$, and $K$ is the number of concepts. -->

<!-- ## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+

### Setup
```bash
# Clone the repository
git clone [repository-url]
cd cbm

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt

# Setup LLaVA submodule
cd ../LLaVA
pip install -e .
 -->

## File Structure
```
cbm/
├── main_pipeline_train.sh            # Training pipeline script
├── main_pipeline_test.sh             # Testing pipeline script
├── feature_extract.py                # COCO feature extraction
├── ms_coco_training_data_trainsform.py # Training data transformation
├── pope_test_data_trainsform.py      # Pope dataset preparation
├── ce_train.sh                       # CBM training script
├── ce_test.py                        # CBM evaluation
├── vqa_feature_extract.py            # VQA feature extraction
├── chair_test_data_trainsform.py     # CHAIR dataset preparation
├── hal_feature_extract.py            # HAL feature extraction
└── result/
    └── plot_offline.py               # Result visualization

LLaVA/
├── ce_datagen.sh                     # LLaVA feature generation for training
├── run-pope-cbm.sh                   # Pope evaluation with LLaVA
├── run-chair-cbm.sh                  # CHAIR evaluation with LLaVA
└── vqa_datagen-v1.py                 # VQA feature generation
```


## Training Pipeline (`cbm/main_pipeline_train.sh`)

### Step 1: Feature Extraction
Extract object annotations from COCO2014 dataset:

```bash
python feature_extract.py
```

#### Output Files:
- `object_dict_full.pth`: Full object dictionary with concept annotations

- `raw_result_full.pth`: Raw detection results from object detector

### Step 2 Training Data Construction:

Transform COCO annotations into concept-embodied training data:

```bash
python ms_coco_training_data_trainsform.py
```

#### Output Files:
- `ce_data.pth`: Dataset containing image features and softmaxed concept ground truth

### Step 3 LLaVA Feature Extraction:
Extract multimodal features using LLaVA:
```bash
cd ../LLaVA
./ce_datagen.sh
```

#### Output Files:
- `ce_training.pth`: Information list for CBM training

- `ce_training_label.pth`: Ground truth concept labels

- `ce_training_response.pth`: LLaVA model outputs

### Step 4 LLaVA Feature Extraction:

Train the Concept Bottleneck Model:
```bash
cd ../cbm
./ce_train.sh
```

#### Output File:
- `ce_model_{index}.pth`: Trained CBM model weights


## Testing Pipeline (cbm/main_pipeline_test.sh)

### 1. Pope Dataset Evaluation

### Step 1: Pope Data Annotation
```bash
python pope_test_data_trainsform.py
```

#### Output File: 
- `pope_ce_test_data.pth`

### Step 2: LLaVA Feature Extraction for Pope

```bash
cd ../LLaVA
./run-pope-cbm.sh
```

#### Output Files:
- pope_info_probe_list.pth

- pope_label_list.pth

- pope_output_list.pth

- pope_question_list.pth

- pope_vit_feature_list.pth


#### Step 3: CBM Evaluation on Pope

```bash
cd ../cbm
python ce_test.py
```

## 2. VQA v2 on COCO Validation

```bash
# Feature extraction
python vqa_feature_extract.py

# Data transformation (training_set=False)
python vqa_ms_coco_training_data_trainsform-v1.py

# LLaVA feature generation
cd ../LLaVA
./vqa_datagen-v1.py
```

## 3. CHAIR Open Dataset

```bash
# Data transformation
python chair_test_data_trainsform.py

# LLaVA feature extraction
cd ../LLaVA
./run-chair-cbm.sh

# CBM evaluation
cd ../cbm
python chair_ce_test-v1.py

# K-means clustering analysis
python kmeans.py
```

## 4. HAL Dataset
```bash
# Feature extraction
python hal_feature_extract.py

# Data transformation (training_set=False)
python hal_ms_coco_training_data_trainsform-v1.py

# LLaVA feature generation
cd ../LLaVA
./hal_datagen-v1.py

# CBM evaluation
cd ../cbm
python hal_ce_test-v1.py
```

## 5. Domain Analysis

```bash
# Question type analysis
python vqa_question_type.py

# Domain injection metrics
python domain_inject_metric.py
```

## YOLO Object Dictionary

The pipeline uses the following 80-class YOLO object dictionary:

```json
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
 78: 'hair drier', 79: 'toothbrush'}
```



@inproceedings{huang25image,
  title = "Image difference captioning via adversarial preference optimization",
  author = "Zihan Huang and Junda Wu and Rohan Surana and Tong Yu and David Arbour and Ritwik Sinha and Julian McAuley",
  year = "2025",
  booktitle = "EMNLP"
}

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{huang25image,
  title = "Image difference captioning via adversarial preference optimization",
  author = "Zihan Huang and Junda Wu and Rohan Surana and Tong Yu and David Arbour and Ritwik Sinha and Julian McAuley",
  year = "2025",
  booktitle = "EMNLP"
}
```
