"""
Training YOLOv8 untuk Deteksi Kecurangan Ujian
"""
import os
from ultralytics import YOLO
import torch
from config import *

def train():
    # Pastikan folder ada
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Path ke data.yaml
    data_yaml = os.path.join(DATA_DIR, 'data.yaml')
    
    if not os.path.exists(data_yaml):
        print("Error: data.yaml tidak ditemukan!")
        print("Jalankan prepare_data.py terlebih dahulu")
        return
    
    # Load YOLOv8 pretrained model
    # Pilihan: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
    model = YOLO('yolov8n.pt')  # Mulai dari nano untuk testing
    
    print("=" * 50)
    print("TRAINING YOLOV8 - Cheating Detection")
    print("=" * 50)
    
    # Resolve device: auto -> GPU if available, else CPU
    resolved_device = DEVICE
    if DEVICE == "auto":
        resolved_device = "0" if torch.cuda.is_available() else "cpu"

    # Training
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=10,
        save=True,
        project=CHECKPOINT_DIR,
        name='cheating_detector',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        augment=True,
        device=resolved_device,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Best model: {CHECKPOINT_DIR}/cheating_detector/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train()
