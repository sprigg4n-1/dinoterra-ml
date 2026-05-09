import os
import subprocess
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

if isinstance(dino_classes_raw, dict):
    dino_classes = [dino_classes_raw[str(i)] for i in range(len(dino_classes_raw))]
else:
    dino_classes = dino_classes_raw

print(f"Завантажено класів: {len(dino_classes)}")

# ── Лічильник для перенавчання ────────────────────────────────────────────────
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
    """Для Stage 1 — бінарна модель (/255.0)"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(size)
    img = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def prepare_image_stage2(image_bytes: bytes) -> np.ndarray:
    """Для Stage 2 — ResNet50 preprocess_input"""
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

# ── 1. USER PREDICT ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        img_s1 = prepare_image(image_bytes, IMG_SIZE)
        stage1_prob = float(model_binary.predict(img_s1, verbose=0)[0][0])
        is_dino = stage1_prob >= 0.5

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


# ── 2. ADMIN PREDICT ──────────────────────────────────────────────────────────
@app.post("/admin/predict")
async def admin_predict(file: UploadFile = File(...)):
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


# ── 4. RETRAIN TRIGGER (адмін) ────────────────────────────────────────────────
@app.post("/retrain_trigger")
async def retrain_trigger():
    global pending_retrain_count, is_retraining

    if is_retraining:
        return {
            "status": "Перенавчання вже запущено",
            "pending_count": pending_retrain_count,
        }

    if pending_retrain_count < 10:
        return {
            "status": "Замало зразків",
            "pending_count": pending_retrain_count,
            "required": 10,
        }

    # Запускаємо retrain.py у фоні
    is_retraining = True
    subprocess.Popen(["python", "retrain.py"])

    return {
        "status": "Перенавчання запущено",
        "pending_count": pending_retrain_count,
    }


# ── 5. RETRAIN DONE ───────────────────────────────────────────────────────────
@app.post("/retrain_done")
async def retrain_done():
    global pending_retrain_count, is_retraining
    pending_retrain_count = 0
    is_retraining = False

    return {
        "status": "Перенавчання завершено",
        "pending_retrain_count": pending_retrain_count,
    }


# ── 6. RETRAIN STATUS ─────────────────────────────────────────────────────────
@app.get("/retrain_status")
async def retrain_status():
    """Адмін може перевірити статус перенавчання"""
    return {
        "is_retraining": is_retraining,
        "pending_count": pending_retrain_count,
        "should_retrain": pending_retrain_count >= RETRAIN_THRESHOLD,
    }


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)