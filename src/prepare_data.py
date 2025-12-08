"""
Prepare dataset untuk YOLOv8 training
"""
import os
import shutil
import random
from config import *

def prepare_yolo_dataset():
    """Siapkan struktur folder YOLO dari dataset asli"""
    
    # Buat struktur folder
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATA_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, split, 'labels'), exist_ok=True)
    
    # Kumpulkan semua pair image-label
    all_data = []
    
    for scenario in os.listdir(DATASET_DIR):
        scenario_path = os.path.join(DATASET_DIR, scenario)
        if not os.path.isdir(scenario_path):
            continue
        
        # Cari semua file jpg
        for f in os.listdir(scenario_path):
            if f.lower().endswith('.jpg'):
                img_path = os.path.join(scenario_path, f)
                txt_path = os.path.join(scenario_path, f.replace('.jpg', '.txt'))
                
                if os.path.exists(txt_path):
                    all_data.append({
                        'image': img_path,
                        'label': txt_path,
                        'name': f.replace('.jpg', '')
                    })
    
    print(f"Total image-label pairs: {len(all_data)}")
    
    # Shuffle dan split
    random.seed(SEED)
    random.shuffle(all_data)
    
    n_total = len(all_data)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    
    splits = {
        'train': all_data[:n_train],
        'val': all_data[n_train:n_train+n_val],
        'test': all_data[n_train+n_val:]
    }
    
    # Copy files
    for split, data_list in splits.items():
        print(f"\nCopying {split}...")
        for item in data_list:
            # Copy image
            dst_img = os.path.join(DATA_DIR, split, 'images', item['name'] + '.jpg')
            shutil.copy2(item['image'], dst_img)
            
            # Copy label
            dst_label = os.path.join(DATA_DIR, split, 'labels', item['name'] + '.txt')
            shutil.copy2(item['label'], dst_label)
        
        print(f"  {split}: {len(data_list)} samples")
    
    # Buat data.yaml
    create_data_yaml()
    
    print("\n✅ Dataset prepared successfully!")
    print(f"   Location: {DATA_DIR}")

def create_data_yaml():
    """Buat file konfigurasi YOLO"""
    yaml_content = f"""# Cheating Detection Dataset
path: {DATA_DIR}
train: train/images
val: val/images
test: test/images

# Classes (13 total)
names:
  0: head_down
  1: head_up
  2: head_left
  3: head_right
  4: head_center
  5: eye_center
  6: eye_left
  7: eye_right
  8: eye_up
  9: eye_down
  10: eye_closed
  11: lip_closed
  12: lip_open

nc: 13
"""
    
    yaml_path = os.path.join(DATA_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated: {yaml_path}")

if __name__ == "__main__":
    print("=" * 50)
    print("PREPARE YOLO DATASET")
    print("=" * 50)
    prepare_yolo_dataset()
