import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Suppress DecompressionBombWarning for large images
Image.MAX_IMAGE_PIXELS = None

# --- 1. Dataset Class ---
class HouseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.paths = dataframe['image_path'].values
        self.labels = dataframe['median_house_value'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            # Fallback for missing or corrupt files
            img = Image.new('RGB', (128, 128), color='black')
        
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float)

# --- 2. CNN Architecture ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 128 -> 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            nn.Flatten()
        )
        # 32 channels * 32px * 32px = 32768 input features
        self.fc = nn.Linear(32 * 32 * 32, 64) 
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(self.conv_stack(x))
        return self.out(torch.relu(x))

    def extract(self, x):
        """Used to get the 64-dimensional feature vector for fusion."""
        return self.fc(self.conv_stack(x))

# --- 3. Helper Functions ---
def prepare_tabular(data_frame, vectorizer=None, fit=False):
    features = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms', 
        'total_bedrooms', 'population', 'households', 'median_income', 
        'bedrooms', 'sq_ft'
    ]
    dicts = data_frame[features].to_dict('records')
    if fit:
        return vectorizer.fit_transform(dicts), data_frame['median_house_value']
    return vectorizer.transform(dicts), data_frame['median_house_value']

def get_cnn_features(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats.append(model.extract(imgs).cpu().numpy())
    return np.vstack(feats)

# --- 4. Main Execution ---
def main():
    # Setup directory for saving models
    if not os.path.exists('model'):
        os.makedirs('model')

    # Load Data
    print("Loading data...")
    df = pd.read_csv('data/dataset.csv')
    df.fillna(0, inplace=True)

    # Split Data
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

    # Prepare Tabular Data
    dv = DictVectorizer(sparse=False)
    X_train, y_train = prepare_tabular(df_train, dv, fit=True)
    X_val, y_val = prepare_tabular(df_val, dv)

    # Image Transforms & Loaders
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # shuffle=False is critical here so features align with X_train/X_val rows
    train_loader = DataLoader(HouseDataset(df_train, transform), batch_size=32, shuffle=False)
    val_loader = DataLoader(HouseDataset(df_val, transform), batch_size=32, shuffle=False)

    # Initialize CNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    cnn = CNN().to(device)

    # Feature Extraction
    print("Extracting CNN features from images...")
    train_img_feats = get_cnn_features(cnn, train_loader, device)
    val_img_feats = get_cnn_features(cnn, val_loader, device)

    # Fusion (Tabular + Image)
    print("Combining features and training XGBoost...")
    X_train_fused = np.hstack([X_train, train_img_feats])
    X_val_fused = np.hstack([X_val, val_img_feats])

    dtrain = xgb.DMatrix(X_train_fused, label=y_train)
    dval = xgb.DMatrix(X_val_fused, label=y_val)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 1
    }
    
    fused_model = xgb.train(params, dtrain, num_boost_round=50)

    # Prediction & Scoring
    y_pred = fused_model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'Final Fused RMSE: {rmse:.2f}')

    # --- 5. Save Artifacts ---
    print("Saving models and vectorizer...")
    # Save XGBoost Fusion Model
    joblib.dump(fused_model, 'model/fusion_model.bin')
    
    # Save DictVectorizer (Required for API to process new tabular data)
    joblib.dump(dv, 'model/dv.bin')
    
    # Save CNN (Required for API to extract features from new images)
    torch.save(cnn, 'model/cnn.pth')
    
    print("All artifacts saved to model/ directory.")

if __name__ == "__main__":
    main()