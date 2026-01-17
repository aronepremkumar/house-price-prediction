import io
import joblib
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, UploadFile, Form
from PIL import Image
from torchvision import transforms
from sklearn.feature_extraction import DictVectorizer

# --- 1. CNN Architecture Definition ---
# Required so torch.load can reconstruct the model structure
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 32 * 32, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(self.conv_stack(x))
        return self.out(torch.relu(x))

    def feature_extractor(self, x):
        return self.fc(self.conv_stack(x))

# --- 2. Initialize App and Load Models ---
app = FastAPI(title="House Price Prediction API")

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved assets
fusion_model = joblib.load('model/fusion_model.bin')
dv = joblib.load('model/dv.bin')

# Load CNN (assumes model was saved via torch.save(model, 'path'))
# Note: If you saved via state_dict, use: cnn.load_state_dict(torch.load(...))
try:
    cnn = torch.load('model/cnn.pth', map_location=device)
except Exception:
    # Fallback if you only have the class locally
    cnn = CNN().to(device)
    print("Warning: cnn.pth not found, using uninitialized weights.")

cnn.eval()

# Image Preprocessing (Matches training dimensions)
transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is online", "model": "Multi-Modal Fusion (XGBoost + CNN)"}

@app.post("/predict")
async def predict(
    image: UploadFile = None,
    longitude: float = Form(...),
    latitude: float = Form(...),
    housing_median_age: float = Form(...),
    total_rooms: float = Form(...),
    total_bedrooms: float = Form(...),
    population: float = Form(...),
    households: float = Form(...),
    median_income: float = Form(...),
):
    # Prepare Tabular Data
    tabular = {
        'longitude': longitude, 
        'latitude': latitude, 
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms, 
        'total_bedrooms': total_bedrooms, 
        'population': population,
        'households': households, 
        'median_income': median_income,
        'bedrooms': total_bedrooms / households if households > 0 else 0,
        'sq_ft': total_rooms * 200
    }
    
    # Transform using fitted DictVectorizer
    tabular_encoded = dv.transform([tabular])

    # Handle Image Data
    if image:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cnn_feats = cnn.feature_extractor(img_tensor).cpu().numpy().flatten()
    else:
        # Default to zeros if no image is provided
        cnn_feats = np.zeros(64)

    # Combine (Fusion)
    input_vec = np.hstack((tabular_encoded.flatten(), cnn_feats))
    
    # XGBoost Prediction
    dmatrix = xgb.DMatrix([input_vec])
    price = fusion_model.predict(dmatrix)[0]

    return {
        "predicted_price": float(price),
        "currency": "USD"
    }