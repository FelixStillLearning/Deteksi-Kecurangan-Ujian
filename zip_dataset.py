"""
Zip dataset untuk upload ke Colab
"""
import os
import zipfile
from src.config import DATA_DIR, BASE_DIR

def zip_dataset():
    zip_path = os.path.join(BASE_DIR, 'dataset.zip')
    
    print("Creating dataset.zip...")
    print(f"Source: {DATA_DIR}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, BASE_DIR)
                zipf.write(file_path, arcname)
                
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n✅ Created: {zip_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print("\nUpload file ini ke Google Colab!")

if __name__ == "__main__":
    zip_dataset()
