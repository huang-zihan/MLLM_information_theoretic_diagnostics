# Concept Bottleneck Models for Multimodal Understanding

[Paper Link](https://openreview.net/pdf?id=pQm66IPmeE)

## File Structure
```
MLLM_information_theoretic_diagnostics/
│
├── cbm/                           # Concept Bottleneck Model pipeline
│   ├── main_pipeline_train.sh     # Full training pipeline
│   ├── main_pipeline_test.sh      # Full testing & evaluation pipeline
│   ├── feature_extract.py        # COCO2014 object annotation extraction (YOLO)
│   ├── ms_coco_training_data_trainsform.py  # Training data construction
│   ├── ce_train.py               # CBM training
│   ├── ce_test.py               # Core CBM evaluation (Pope, etc.)
│   │
│   ├── pope_test_data_trainsform.py      # Pope dataset annotation
│   ├── chair_test_data_trainsform.py     # CHAIR dataset annotation
│   ├── hal_feature_extract.py           # HAL feature extraction
│   ├── hal_ms_coco_training_data_trainsform.py
│   ├── vqa_feature_extract.py           # VQA v2 feature extraction
│   ├── vqa_ms_coco_training_data_trainsform.py
│   ├── coco_caption_feature_extract.py  # COCO Caption feature extraction
│   ├── coco_caption_training_data_trainsform.py
│   ├── aokvqa_test_data_trainsform.py   # AOK-VQA test data transformation
│   │
│   ├── chair_ce_test.py          # CHAIR evaluation
│   ├── hal_ce_test.py            # HAL evaluation
│   ├── vqa_ce_test.py            # VQA v2 evaluation
│   ├── aokvqa_ce_test.py         # AOK-VQA evaluation
│   ├── coco_caption_ce_test.py   # COCO Caption evaluation
│   ├── domain_ce_test.py         # Domain shift evaluation
│   │
│   ├── kmeans.py                 # K-means clustering analysis
│   ├── vqa_question_type.py      # Question type analysis (VQA)
│   │
│   ├── similarity/               # Concept injection & information metrics
│   │   ├── domain_inject_metric.py
│   │   ├── hal_visual_inject_metric.py
│   │   ├── inject_metric.py
│   │   ├── visual_inject_metric.py
│   │   ├── text_inject_metric.py
│   │   ├── similarity.py
│   │   └── metric2_plot_offline.py  # Result visualization
│   │
│   └── result/                   # Output plots (generated)
│       └── plot_offline.py
│
└── LLaVA/                        # LLaVA/Qwen feature extraction & inference
    ├── env.yml                  # Conda environment
    ├── ce_datagen.sh            # Training feature generation (CBM training)
    ├── ce_datagen.py
    ├── run-pope-cbm.sh          # Pope evaluation with LLaVA/Qwen
    ├── run-pope-cbm.py
    ├── run-chair-cbm.sh         # CHAIR evaluation
    ├── run-chair-cbm.py
    ├── run-aokvqa-cbm.sh        # AOK-VQA evaluation
    ├── run-aokvqa-cbm.py
    ├── hal_datagen.sh           # HAL dataset feature generation
    ├── hal_datagen.py
    ├── vqa_datagen.sh           # VQA v2 feature generation
    ├── vqa_datagen.py
    ├── vqa_datagen-intervene.py # Intervention-based feature generation
    ├── coco_caption_datagen.sh  # COCO Caption feature generation
    ├── coco_caption_datagen.py
    ├── classify.sh              # Final Fv calculation (classification probe)
    └── classify.py
```


## Training Pipeline (`cbm/main_pipeline_train.sh`)

### Step 1: COCO Object Annotation Extraction
Extract YOLO object detections from COCO2014 training set.


```bash
cd cbm
python feature_extract.py
```

#### Output Files:
- `object_dict_full.pth`: Full object dictionary with concept annotations

- `raw_result_full.pth`: Raw detection results from object detector

### Step 2 Construct Concept‑Embodied Training Data

Transform COCO annotations into a dataset with image features and softmax‑normalized concept ground truths.

```bash
python ms_coco_training_data_trainsform.py
```

#### Output Files:
- `ce_data.pth`: Dataset containing image features and softmaxed concept ground truth

### Step 3 Extract LLaVA/Qwen Features
Run LLaVA or Qwen on the constructed dataset to obtain multimodal representations.
```bash
cd ../LLaVA
./ce_datagen.sh          # contains branch for Qwen
```

#### Output Files:
- `ce_training.pth`: Information list for CBM training

- `ce_training_label.pth`: Ground truth concept labels

- `ce_training_response.pth`: LLaVA/Qwen model outputs

### Train Concept Bottleneck Model

Train the Concept Bottleneck Model:
```bash
cd ../cbm
./ce_train.sh
```

#### Output File:
- `ce_model_{index}.pth`: Trained CBM model weights


## Extract Hidden on Other Datasets and Evaluate  (cbm/main_pipeline_test.sh)

### Pope Dataset Evaluation

```bash
# Data annotation (using object_dict from training)
python pope_test_data_trainsform.py          # → pope_ce_test_data.pth

# LLaVA/Qwen feature extraction
cd ../LLaVA
./run-pope-cbm.sh                            # → pope_*_list.pth files

# CBM evaluation
cd ../cbm
python ce_test.py
```

### AOK-VQA Dataset

```bash
python aokvqa_test_data_trainsform.py
cd ../LLaVA
./run-aokvqa-cbm.sh
cd ../cbm
python aokvqa_ce_test.py
```

### CHAIR Dataset

```bash
python chair_test_data_trainsform.py
cd ../LLaVA
./run-chair-cbm.sh
cd ../cbm
python chair_ce_test.py
python kmeans.py          # clustering analysis
```

### HAL Dataset
```bash
python hal_feature_extract.py
python hal_ms_coco_training_data_trainsform.py   # training_set=False
cd ../LLaVA
./hal_datagen.sh
cd ../cbm
python hal_ce_test.py
python kmeans.py
```

### VQA v2 (COCO validation)
```bash
python vqa_feature_extract.py
python vqa_ms_coco_training_data_trainsform.py   # training_set=False
cd ../LLaVA
./vqa_datagen.sh
cd ../cbm
python vqa_ce_test.py
python kmeans.py
python vqa_question_type.py
```

### COCO Caption
```bash
python coco_caption_feature_extract.py
python coco_caption_training_data_trainsform.py
cd ../LLaVA
./coco_caption_datagen.sh
cd ../cbm
python coco_caption_ce_test.py
python kmeans.py
```

## Domain Analysis
### Question type analysis
```bash
python vqa_question_type.py
```

### Domain shift evaluation
```bash
python domain_ce_test.py
```

## Concept Injection Metrics

### Fₜ, injection scores
```bash
cd similarity
python domain_inject_metric.py
python inject_metric.py
python visual_inject_metric.py
python text_inject_metric.py
python hal_visual_inject_metric.py
```

## Concept Injection Metrics
```bash
python result/metric2_plot_offline.py   # offline plots from saved results
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

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{huang2025traceable,
  title={Traceable and Explainable Multimodal Large Language Models: An Information-Theoretic View},
  author={Huang, Zihan and Wu, Junda and Surana, Rohan and Jain, Raghav and Yu, Tong and Addanki, Raghavendra and Arbour, David and Kim, Sungchul and McAuley, Julian},
  booktitle={Second Conference on Language Modeling},
  year={2025}
}
```
