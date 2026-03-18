# YOLO11n-CBAM-PlantVillage
YOLO11n-CBAM: A Lightweight Plant Disease Detection Model

## Overview
This repository contains the code, model weights, and experimental results of **YOLO11n-CBAM**, a lightweight plant disease detection model trained on the PlantVillage dataset. The model is optimized for real-time detection on resource-constrained devices (e.g., laptops with RTX 3060 GPU).

## Key Features
- **Lightweight**: Only 2.59M parameters, 6.4 GFLOPs
- **High Performance**: mAP@0.5 = 0.9930 on PlantVillage dataset, inference speed = 217.39 FPS (4.6ms per image)
- **CBAM Attention**: Integrates CBAM (Convolutional Block Attention Module) to enhance feature extraction for subtle plant disease lesions

## Environment
- Python 3.12.4
- PyTorch 2.7.1+cu118
- Ultralytics 8.3.163
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6144MiB)

## Core Files
- `eval_model.py`: Evaluation script to compute model metrics (mAP@0.5, Precision, Recall, FPS, etc.)
- `step3_train_cbam.py`: Training script for YOLO11n-CBAM
- `last.pt`: Trained model weights (YOLO11n-CBAM)
- `results.csv`: Training metrics (loss, mAP@0.5)
- `results.png`: Training curve (loss & mAP@0.5)

## Usage
### Evaluate the Model
```bash
python eval_model.py
Train the Model
bash
运行
python step3_train_cbam.py
Experimental Results
表格
Model	Params (M)	GFLOPs	mAP@0.5	Precision	Recall	Inference Speed (FPS)
YOLO11n (Baseline)	2.56	6.2	0.9780	0.9750	0.9720	120.48
YOLO11n-CBAM (Ours)	2.59	6.4	0.9930	0.9920	0.9900	217.39
Data Availability
The PlantVillage dataset is publicly available at: https://plantvillage.psu.edu/
Citation
If you use this code or model in your research, please cite our paper (to be published in IEEE Access).
