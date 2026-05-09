import os
import uvicorn
import numpy as np
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess, decode_predictions
from tensorflow.keras import optimizers
from PIL import Image
import io

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import (
    IMG_SIZE, IMG_SIZE_BIG,
    LEARNING_RATE,
    MODEL_PATH, MODEL_PATH_DINO_CLASS,
    DINO_CLASSES_PATH,
    NEW_DINO_FOLDER,
    RETRAIN_THRESHOLD,
)

# ── Завантаження моделей ──────────────────────────────────────────────────────
print("Завантаження моделей...")
model_binary   = load_model(MODEL_PATH)
model_dino     = load_model(MODEL_PATH_DINO_CLASS)
model_non_dino = ResNet50(weights="imagenet")
print("Всі моделі завантажено!")

with open(DINO_CLASSES_PATH, "r", encoding="utf-8") as f:
    dino_classes_raw = json.load(f)

# Конвертуємо словник в список якщо потрібно
if isinstance(dino_classes_raw, dict):
    dino_classes = [dino_classes_raw[str(i)] for i in range(len(dino_classes_raw))]
else:
    dino_classes = dino_classes_raw

print(f"Завантажено класів: {len(dino_classes)}")

# ── Лічильник для перенавчання ────────────────────────────────────────────────
pending_retrain_count = 0

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="DinoTerra ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Хелпери ───────────────────────────────────────────────────────────────────
def prepare_image(image_bytes: bytes, size: tuple) -> np.ndarray:
    """Для Stage 1 — бінарна модель (/255.0)"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(size)
    img = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def prepare_image_stage2(image_bytes: bytes) -> np.ndarray:
    """Для Stage 2 — ResNet50 preprocess_input як у notebook"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE_BIG)
    img = np.array(image).astype("float32")
    img = np.expand_dims(img, axis=0)
    return resnet_preprocess(img)

def prepare_image_resnet(image_bytes: bytes) -> np.ndarray:
    """Для non-dino ResNet50"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE_BIG)
    img = np.array(image).astype("float32")
    img = np.expand_dims(img, axis=0)
    return resnet_preprocess(img)

def save_for_retrain(image_bytes: bytes, correct_class: str, prediction_id: str):
    """Зберігає фото у папку для перенавчання"""
    target_folder = os.path.join(NEW_DINO_FOLDER, correct_class)
    os.makedirs(target_folder, exist_ok=True)
    path = os.path.join(target_folder, f"{prediction_id}.jpg")
    with open(path, "wb") as f:
        f.write(image_bytes)

# ── 1. USER PREDICT ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Для користувача:
      - Stage 1 (150x150, /255.0) → динозавр чи ні
      - Stage 2 (224x224, resnet_preprocess) → топ-3 види
    Фото НЕ зберігається — все в MongoDB на беці.
    """
    try:
        image_bytes = await file.read()

        # Stage 1 — бінарна
        img_s1 = prepare_image(image_bytes, IMG_SIZE)
        stage1_prob = float(model_binary.predict(img_s1, verbose=0)[0][0])
        is_dino = stage1_prob >= 0.5

        # Stage 2 — топ-3
        if is_dino:
            img_s2 = prepare_image_stage2(image_bytes)
            stage2_preds = model_dino.predict(img_s2, verbose=0)[0]
            top3_indices = np.argsort(stage2_preds)[::-1][:3]
            top3 = [
                {
                    "rank": i + 1,
                    "species": dino_classes[int(idx)],
                    "confidence": round(float(stage2_preds[idx]), 4),
                }
                for i, idx in enumerate(top3_indices)
            ]
        else:
            img_resnet = prepare_image_resnet(image_bytes)
            preds = model_non_dino.predict(img_resnet, verbose=0)
            decoded = decode_predictions(preds, top=3)[0]
            top3 = [
                {
                    "rank": i + 1,
                    "species": item[1],
                    "confidence": round(float(item[2]), 4),
                }
                for i, item in enumerate(decoded)
            ]

        prediction_id = datetime.now().strftime("%Y%m%d_%H%M%S%f")

        return {
            "prediction_id": prediction_id,
            "is_dinosaur": is_dino,
            "stage1_probability": round(stage1_prob, 4),
            "top3": top3,
        }

    except Exception as e:
        print(f"ПОМИЛКА /predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. FEEDBACK ───────────────────────────────────────────────────────────────
@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    prediction_id: str = Form(...),
    correct_class: str = Form(...),
):
    """
    Отримує фото від Node.js (з MongoDB) і зберігає для перенавчання.
    Викликається тільки коли модель помилилась і юзер вказав правильний клас.
    """
    global pending_retrain_count

    try:
        if correct_class not in dino_classes:
            raise HTTPException(status_code=400, detail=f"Невідомий клас: {correct_class}")

        image_bytes = await file.read()
        save_for_retrain(image_bytes, correct_class, prediction_id)
        pending_retrain_count += 1

        return {
            "status": "Збережено для перенавчання",
            "correct_class": correct_class,
            "pending_retrain_count": pending_retrain_count,
            "should_retrain": pending_retrain_count >= RETRAIN_THRESHOLD,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ПОМИЛКА /feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. ADMIN PREDICT ──────────────────────────────────────────────────────────
@app.post("/admin/predict")
async def admin_predict(file: UploadFile = File(...)):
    """
    Для адміна:
      - Якщо динозавр → топ-1 вид
      - Якщо не динозавр → що це за річ (топ-1)
    """
    try:
        image_bytes = await file.read()

        img_s1 = prepare_image(image_bytes, IMG_SIZE)
        stage1_prob = float(model_binary.predict(img_s1, verbose=0)[0][0])
        is_dino = stage1_prob >= 0.5

        if is_dino:
            img_s2 = prepare_image_stage2(image_bytes)
            stage2_preds = model_dino.predict(img_s2, verbose=0)[0]
            class_idx = int(np.argmax(stage2_preds))

            return {
                "is_dinosaur": True,
                "stage1_probability": round(stage1_prob, 4),
                "species": dino_classes[class_idx],
                "confidence": round(float(stage2_preds[class_idx]), 4),
            }

        else:
            img_resnet = prepare_image_resnet(image_bytes)
            preds = model_non_dino.predict(img_resnet, verbose=0)
            decoded = decode_predictions(preds, top=1)[0][0]

            return {
                "is_dinosaur": False,
                "stage1_probability": round(stage1_prob, 4),
                "species": decoded[1],
                "confidence": round(float(decoded[2]), 4),
            }

    except Exception as e:
        print(f"ПОМИЛКА /admin/predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. TRAIN SINGLE (адмін) ───────────────────────────────────────────────────
@app.post("/train_single")
async def train_single(
    file: UploadFile = File(...),
    correct_label: int = Form(...),
):
    """Донавчає stage1 модель на одному прикладі"""
    try:
        image_bytes = await file.read()
        img = prepare_image(image_bytes, IMG_SIZE)
        y = np.array([correct_label], dtype="float32")

        for layer in model_binary.layers[:-3]:
            layer.trainable = False

        model_binary.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(LEARNING_RATE * 0.25),
            metrics=["accuracy"]
        )

        history = model_binary.fit(img, y, epochs=1, verbose=0)
        model_binary.save(MODEL_PATH)

        return {
            "status": "Stage1 модель донавчено",
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
        }

    except Exception as e:
        print(f"ПОМИЛКА /train_single: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. RETRAIN TRIGGER (адмін) ────────────────────────────────────────────────
@app.post("/retrain_trigger")
async def retrain_trigger():
    """Перевіряє чи є достатньо зразків для перенавчання"""
    global pending_retrain_count

    if pending_retrain_count < 10:
        return {
            "status": "Замало зразків",
            "pending_count": pending_retrain_count,
            "required": 10,
        }

    return {
        "status": "Готово до перенавчання",
        "pending_count": pending_retrain_count,
        "message": "Запусти retrain.py для перенавчання моделі",
    }


# ── 6. RETRAIN DONE ───────────────────────────────────────────────────────────
@app.post("/retrain_done")
async def retrain_done():
    """Викликається після успішного перенавчання — скидає лічильник"""
    global pending_retrain_count
    pending_retrain_count = 0

    return {
        "status": "Лічильник скинуто",
        "pending_retrain_count": pending_retrain_count,
    }


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)