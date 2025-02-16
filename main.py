

import os
import uuid
import io
import cv2
import json
import torch
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Import segmentation function from image_highlight.py
from image_highlight import segment_and_highlight

# Initialize FastAPI app
app = FastAPI()

# # Load the model and image processor from the saved directory
# model_dir = "./saved_model"
# image_processor = AutoImageProcessor.from_pretrained(model_dir)
# model = AutoModelForImageClassification.from_pretrained(model_dir)

# Load the model and image processor
repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
image_processor = AutoImageProcessor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)


# Define class names
class_names = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa', 'Hailey-Hailey Disease',
    'Herpes Simplex', 'Impetigo', 'Larva Migrans', 'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid',
    'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma', 'Molluscum Contagiosum',
    'Mycosis Fungoides', 'Neurofibromatosis', 'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis',
    'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis', 'Tinea Corporis', 'Tinea Nigra', 'Tungiasis',
    'actinic keratosis', 'dermatofibroma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis',
    'squamous cell carcinoma', 'vascular lesion'
]

# Load disease information from JSON file
with open('disease_info.json', 'r') as f:
    disease_info = json.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image into a PIL Image
        pil_image = Image.open(io.BytesIO(await file.read()))
        
        # Prepare image for prediction
        encoding = image_processor(pil_image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        
        # Get prediction and probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
        predicted_class_idx = logits.argmax(-1).item()
        prediction = class_names[predicted_class_idx]
        disease_data = disease_info.get(prediction, {"description": "Information not available."})
        top3_probabilities = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)[:3]
        
        # Save the uploaded image temporarily (so the segmentation function can read it from disk)
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        pil_image.save(temp_filename)
        
        # Call the segmentation and highlighting function from image_highlight.py
        highlighted_cv_image = segment_and_highlight(temp_filename, k=2)
        
        # Remove the temporary file
        os.remove(temp_filename)
        
        # Encode the highlighted image as PNG and then as a base64 string
        success, buffer = cv2.imencode('.png', highlighted_cv_image)
        if not success:
            raise Exception("Could not encode image")
        highlighted_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Build the JSON response with prediction, probabilities, disease info, and the highlighted image data
        response_data = {
            "prediction": prediction,
            "top_probabilities": {name: prob for name, prob in top3_probabilities},
            **disease_data,
            "highlighted_image": highlighted_image_b64  # The client can decode this base64 PNG image
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing image: {str(e)}"}, status_code=400)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
