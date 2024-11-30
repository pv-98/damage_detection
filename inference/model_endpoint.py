# inference/app.py

import io
import os
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
import uvicorn

# Import your model and configuration
from models.get_model import get_model
import config
#from lensor_new.models import get_model

def get_model_path():

    cloud_model_path = os.getenv("MODEL_PATH", None)

    if cloud_model_path:
        return cloud_model_path
    else: 
        return config.MODEL_PATH
    

# Load the trained model
def load_model(num_classes, model_path, device):
    model = get_model('faster_rcnn', num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


CLASS_NAMES = [
    "none",
    "minor-dent",
    "minor-scratch",
    "moderate-broken",
    "moderate-dent",
    "moderate-scratch",
    "severe-broken",
    "severe-dent",
    "severe-scratch"
]

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
num_classes = config.NUM_CLASSES  # Assuming NUM_CLASSES is defined in config.py
model_path = get_model_path()     # Path to your trained model
model = load_model(num_classes, model_path, device)

# Define the FastAPI app
app = FastAPI()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).to(device)
    # Perform inference
    with torch.no_grad():
        prediction = model([img_tensor])
    # Process output
    output = []
    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score > 0.5:
            output.append({
                'box': [float(coord) for coord in box],
                'label': CLASS_NAMES[label],
                'score': float(score)
            })
    if output:
        return {'predictions': output}
    
    else:
        return {'message': 'Detected damage has a confidence less than threshold' }

# Run the app (if running directly)
if __name__ == "__main__":
    uvicorn.run("inference.app:app", host='0.0.0.0', port=8000)
