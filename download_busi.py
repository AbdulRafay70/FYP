import os
import shutil
import kagglehub

# Define the target dataset directory
dest_dir = r"e:\FYP\datasets\BUSI"

# Create the folder if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)
print(f"Dataset folder ensured at: {dest_dir}")

# Download latest version using kagglehub
print("Downloading dataset using kagglehub...")
cache_path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")

print(f"Dataset successfully downloaded to cache: {cache_path}")
print("Moving files to the target dataset folder...")

# Copy files from the cache directory to the target directory
for item in os.listdir(cache_path):
    source = os.path.join(cache_path, item)
    destination = os.path.join(dest_dir, item)
    
    if os.path.isdir(source):
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)

print("Done! Dataset is ready to use.")
