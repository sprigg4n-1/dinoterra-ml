import os
import subprocess
import uvicorn
import numpy as np
import json
import requests
from datetime import datetime
from fastapi import FastAPI, UploadFile, Request, File, Form, HTTPException
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
    RETRAIN_THRESHOLD,
)

NODE_API_URL = "http://localhost:9000"

# ── Шляхи до нових моделей (тимчасові під час ретрену) ───────────────────────
MODEL_PATH_NEW         = "models/stage1_binary_new.keras"
MODEL_PATH_DINO_NEW    = "models/stage2_dino_species_new.keras"

# ── Завантаження моделей ──────────────────────────────────────────────────────
print("Завантаження моделей...")
active_model_binary = load_model(MODEL_PATH)
active_model_dino   = load_model(MODEL_PATH_DINO_CLASS)
model_non_dino      = ResNet50(weights="imagenet")
print("Всі моделі завантажено!")

with open(DINO_CLASSES_PATH, "r", encoding="utf-8") as f:
    dino_classes_raw = json.load(f)

if isinstance(dino_classes_raw, dict):
    dino_classes = [dino_classes_raw[str(i)] for i in range(len(dino_classes_raw))]
else:
    dino_classes = dino_classes_raw

print(f"Завантажено класів: {len(dino_classes)}")

# ── Стан ретрену ──────────────────────────────────────────────────────────────
pending_retrain_count = 0
is_retraining = False

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
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(size)
    img = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def prepare_image_stage2(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE_BIG)
    img = np.array(image).astype("float32")
    img = np.expand_dims(img, axis=0)
    return resnet_preprocess(img)

def prepare_image_resnet(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE_BIG)
    img = np.array(image).astype("float32")
    img = np.expand_dims(img, axis=0)
    return resnet_preprocess(img)

def reload_dino_classes():
    """Перезавантажує класи динозаврів після ретрену"""
    global dino_classes
    with open(DINO_CLASSES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        dino_classes = [raw[str(i)] for i in range(len(raw))]
    else:
        dino_classes = raw
    print(f"Класи оновлено: {len(dino_classes)}")

# ── 1. USER PREDICT ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        img_s1 = prepare_image(image_bytes, IMG_SIZE)
        stage1_prob = float(active_model_binary.predict(img_s1, verbose=0)[0][0])
        is_dino = stage1_prob >= 0.5

        if is_dino:
            img_s2 = prepare_image_stage2(image_bytes)
            stage2_preds = active_model_dino.predict(img_s2, verbose=0)[0]
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


# ── 2. ADMIN PREDICT ──────────────────────────────────────────────────────────
@app.post("/admin/predict")
async def admin_predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        img_s1 = prepare_image(image_bytes, IMG_SIZE)
        stage1_prob = float(active_model_binary.predict(img_s1, verbose=0)[0][0])
        is_dino = stage1_prob >= 0.5

        if is_dino:
            img_s2 = prepare_image_stage2(image_bytes)
            stage2_preds = active_model_dino.predict(img_s2, verbose=0)[0]
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


# ── 3. TRAIN SINGLE (адмін) ───────────────────────────────────────────────────
@app.post("/train_single")
async def train_single(
    file: UploadFile = File(...),
    correct_label: int = Form(...),
):
    try:
        image_bytes = await file.read()
        img = prepare_image(image_bytes, IMG_SIZE)
        y = np.array([correct_label], dtype="float32")

        for layer in active_model_binary.layers[:-3]:
            layer.trainable = False

        active_model_binary.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(LEARNING_RATE * 0.25),
            metrics=["accuracy"]
        )

        history = active_model_binary.fit(img, y, epochs=1, verbose=0)
        active_model_binary.save(MODEL_PATH)

        return {
            "status": "Stage1 модель донавчено",
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
        }

    except Exception as e:
        print(f"ПОМИЛКА /train_single: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. RETRAIN TRIGGER (адмін) ────────────────────────────────────────────────
@app.post("/retrain_trigger")
async def retrain_trigger(request: Request):
    global is_retraining

    if is_retraining:
        return {"status": "Перенавчання вже запущено"}

    body = await request.json()
    retrain_id = body.get("retrainId", "")

    try:
        response = requests.get(f"{NODE_API_URL}/api/v1/ml/admin/retrain-images")
        data = response.json()
        images = data.get("data", [])
        count = len(images)
    except:
        count = 0

    if count < RETRAIN_THRESHOLD:
        return {
            "status": "Замало зразків",
            "pending_count": count,
            "required": RETRAIN_THRESHOLD,
        }

    is_retraining = True
    subprocess.Popen(["python", "retrain.py", retrain_id])

    return {
        "status": "Перенавчання запущено",
        "pending_count": count,
        "retrainId": retrain_id,
    }


# ── 5. RETRAIN DONE — hot-swap моделей ───────────────────────────────────────
@app.post("/retrain_done")
async def retrain_done():
    global active_model_binary, active_model_dino, is_retraining, pending_retrain_count

    # Завантажуємо нову Stage 1 якщо є
    if os.path.exists(MODEL_PATH_NEW):
        print("Завантаження нової Stage 1 моделі...")
        active_model_binary = load_model(MODEL_PATH_NEW)
        os.replace(MODEL_PATH_NEW, MODEL_PATH)
        print("Stage 1 модель оновлено!")

    # Завантажуємо нову Stage 2 якщо є
    if os.path.exists(MODEL_PATH_DINO_NEW):
        print("Завантаження нової Stage 2 моделі...")
        active_model_dino = load_model(MODEL_PATH_DINO_NEW)
        os.replace(MODEL_PATH_DINO_NEW, MODEL_PATH_DINO_CLASS)
        print("Stage 2 модель оновлено!")

    # Оновлюємо класи якщо змінились
    reload_dino_classes()

    is_retraining = False
    pending_retrain_count = 0

    return {"status": "Перенавчання завершено, модель оновлено"}


# ── 6. RETRAIN STATUS ─────────────────────────────────────────────────────────
@app.get("/retrain_status")
async def retrain_status():
    return {
        "is_retraining": is_retraining,
        "pending_count": pending_retrain_count,
        "should_retrain": pending_retrain_count >= RETRAIN_THRESHOLD,
    }


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)