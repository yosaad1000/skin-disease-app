from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io
import torch

# Initialize FastAPI app
app = FastAPI()

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        image = Image.open(io.BytesIO(await file.read()))
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")

        # Make a prediction
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits

        # Get predicted class
        predicted_class_idx = logits.argmax(-1).item()
        prediction = class_names[predicted_class_idx]

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(content={"error": f"Error processing image: {str(e)}"}, status_code=400)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
