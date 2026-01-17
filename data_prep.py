import pandas as pd
import os
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

# Initialize and authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# 1. Create data directory if it doesn't exist
os.makedirs('data/cubicasa', exist_ok=True)

# 2. Download & Unzip California Housing Prices
print("Downloading California Housing data...")
api.dataset_download_files('camnugent/california-housing-prices', path='data/', unzip=True)
df = pd.read_csv('data/housing.csv')

# Feature Engineering
df['bedrooms'] = df['total_bedrooms'] / df['households']
df['sq_ft'] = df['total_rooms'] * 200 

# 3. Download & Unzip CubiCasa5k images
print("Downloading CubiCasa5k images (this may take a while)...")
api.dataset_download_files('qmarva/cubicasa5k', path='data/cubicasa/', unzip=True)

# 4. RECURSIVE Search for images
# This looks in all subfolders for .jpg and .png files
image_dir = Path('data/cubicasa/')
image_files = list(image_dir.rglob('*.jpg')) + list(image_dir.rglob('*.png'))

# 5. Safety Check: Verify images were found
num_images = len(image_files)
print(f"Total images found: {num_images}")

if num_images == 0:
    print("Error: No images found. Check if the dataset unzipped correctly.")
    df['image_path'] = "no_image_found"
else:
    # Convert Path objects to strings for the CSV
    image_paths = [str(f) for f in image_files]
    # Randomly assign images (sampling with replacement if df is larger than image count)
    df['image_path'] = np.random.choice(image_paths, size=len(df))

# 6. Save the final dataset
df.to_csv('data/dataset.csv', index=False)
print("Successfully prepared: data/dataset.csv with images paired.")