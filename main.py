from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI(title="Food Link AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class_names = [
    "Arborio",
    "Basmati",
    "Ipsala",
    "Jasmine",
    "Karacadag"
]

model_path = hf_hub_download(
    repo_id="Janasree123/foodlink-rice-classifier",
    filename="rice_classifier.keras"
)

rice_model = tf.keras.models.load_model(model_path)

@app.get("/")
def home():
    return {"message": "Food Link AI API is running"}

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((128, 128))

        image_array = tf.keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = rice_model.predict(image_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))

        return {
            "predicted_food": class_names[predicted_index],
            "confidence": round(
                float(predictions[0][predicted_index]) * 100, 2
            )
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

