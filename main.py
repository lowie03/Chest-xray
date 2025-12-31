import io
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

# --- 1. DEFINE MODEL ARCHITECTURE ---
# This must match your notebook exactly
class PneumoniaEfficientNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(PneumoniaEfficientNet, self).__init__()
        # Load backbone
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Replace classifier head
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

# --- 2. GLOBAL VARIABLES ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "improved_pneumonia_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

# Define the exact transforms used in validation/testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. LIFESPAN MANAGER (Load model on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        print(f"Loading model from {MODEL_PATH} on {device}...")
        model = PneumoniaEfficientNet(num_classes=2)
        
        # Load weights
        # map_location ensures it loads on CPU if CUDA is not available on server
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        yield
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Debugging: List files in the directory to help identify naming issues
        print(f"Files in {os.path.dirname(MODEL_PATH)}: {os.listdir(os.path.dirname(MODEL_PATH))}")
        raise e
    finally:
        # Cleanup code if needed
        pass

# --- 4. FASTAPI APP ---
app = FastAPI(title="Pneumonia Detection API", lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Extract probabilities
            normal_prob = probs[0][0].item()
            pneumonia_prob = probs[0][1].item()
            
            # Determine class
            predicted_class = "Pneumonia" if pneumonia_prob > 0.5 else "Normal"
            confidence = max(normal_prob, pneumonia_prob)

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "normal": round(normal_prob, 4),
                "pneumonia": round(pneumonia_prob, 4)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")