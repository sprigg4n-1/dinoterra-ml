import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from PIL import Image
import io

from config import IMG_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_PATH_DINO_CLASS, MODEL_PATH_NON_DINO_CLASS,  DINO_CLASSES_PATH

print("Завантаження моделі...")
model = load_model(MODEL_PATH)
print("Модель завантажено!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ["Не динозавр", "Динозавр"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)

        img = np.array(image).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        label = int(prediction >= 0.5)

        return {
            "class": class_names[label],
            "probability": float(prediction),
            "raw_prediction": float(prediction)
        }

    except Exception as e:
        print(e)
        return {"error": str(e)}


@app.post("/train_single")
async def train_single(
    file: UploadFile = File(...),
    correct_label: int = Form(...)
):
    try:
        # 1. Читаємо картинку
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)

        img = np.array(image).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # 2. Формуємо правильний лейбл
        y = np.array([correct_label], dtype="float32")

        for layer in model.layers[:-3]:
            layer.trainable = False

        # 3. Компілюємо модель з малим learning rate
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(LEARNING_RATE * 0.25),
            metrics=["accuracy"]
        )

        # 4. Донавчання на одному прикладі
        history = model.fit(
            img,
            y,
            epochs=1,
            verbose=0
        )

        # 5. Зберігаємо оновлену модель
        model.save(MODEL_PATH)

        return {
            "status": "Модель успішно донавчено",
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
