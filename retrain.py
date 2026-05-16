import os
import json
import base64
import shutil
import requests
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    IMG_SIZE, IMG_SIZE_BIG,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_PATH, MODEL_PATH_DINO_CLASS,
    DINO_FOLDER,
    DINO_CLASSES_PATH,
    RETRAIN_THRESHOLD,
    NON_DINO_FOLDER
)

NODE_API_URL = "http://localhost:9000"
ML_API_URL   = "http://localhost:8000"

TEMP_DINO_FOLDER     = "dataset/temp_retrain/dino"
TEMP_BINARY_DINO     = "dataset/temp_retrain/binary/dinosaur"
TEMP_BINARY_NON_DINO = "dataset/temp_retrain/binary/not_dinosaur"

# ── Логування ──────────────────────────────────────────────────────────────────
def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

# ── Отримати фото з MongoDB через Node.js API ──────────────────────────────────
def fetch_retrain_images() -> list:
    try:
        response = requests.get(f"{NODE_API_URL}/api/v1/ml/admin/retrain-images")
        data = response.json()
        if data.get("success"):
            return data.get("data", [])
        return []
    except Exception as e:
        log(f"Помилка отримання фото: {e}")
        return []

# ── Зберегти фото тимчасово на диск ───────────────────────────────────────────
def save_temp_images(images: list):
    os.makedirs(TEMP_BINARY_DINO, exist_ok=True)
    os.makedirs(TEMP_BINARY_NON_DINO, exist_ok=True)

    for img_data in images:
        file_b64      = img_data.get("file", "")
        mime_type     = img_data.get("mimeType", "image/jpeg")
        error_type    = img_data.get("errorType")
        correct_class = img_data.get("correctClass")

        if "," in file_b64:
            file_b64 = file_b64.split(",")[1]

        image_bytes = base64.b64decode(file_b64)
        filename = f"{img_data.get('_id', 'img')}.jpg"

        if error_type == "FALSE_POSITIVE":
            path = os.path.join(TEMP_BINARY_NON_DINO, filename)
            with open(path, "wb") as f:
                f.write(image_bytes)

        elif error_type in ("FALSE_NEGATIVE", "WRONG_SPECIES", "NEW_SPECIES"):
            path = os.path.join(TEMP_BINARY_DINO, filename)
            with open(path, "wb") as f:
                f.write(image_bytes)

            if correct_class:
                species_folder = os.path.join(TEMP_DINO_FOLDER, correct_class)
                os.makedirs(species_folder, exist_ok=True)
                path2 = os.path.join(species_folder, filename)
                with open(path2, "wb") as f:
                    f.write(image_bytes)

    log("Тимчасово збережено фото для перенавчання")

# ── Копіювання зображень ───────────────────────────────────────────────────────
def copy_images(src: str, dest: str, recursive: bool = False):
    if not os.path.exists(src):
        return
    os.makedirs(dest, exist_ok=True)

    if recursive:
        for root, _, files in os.walk(src):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    rel = os.path.relpath(root, src)
                    target_dir = os.path.join(dest, rel) if rel != "." else dest
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy2(os.path.join(root, fname), os.path.join(target_dir, fname))
    else:
        for fname in os.listdir(src):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy2(os.path.join(src, fname), os.path.join(dest, fname))

# ── Перенавчання Stage 1 (бінарна) ────────────────────────────────────────────
def retrain_stage1():
    log("=== Перенавчання Stage 1 (бінарна класифікація) ===")

    combined      = "dataset/binary_combined"
    dino_dest     = os.path.join(combined, "dinosaur")
    non_dino_dest = os.path.join(combined, "not_dinosaur")

    os.makedirs(dino_dest, exist_ok=True)
    os.makedirs(non_dino_dest, exist_ok=True)

    copy_images(DINO_FOLDER, dino_dest, recursive=True)
    copy_images(NON_DINO_FOLDER, non_dino_dest, recursive=True)
    copy_images(TEMP_BINARY_DINO, dino_dest)
    copy_images(TEMP_BINARY_NON_DINO, non_dino_dest)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        combined,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        combined,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    if train_gen.samples == 0:
        log("Немає зображень. Пропускаємо.")
        shutil.rmtree(combined)
        return None

    log("Завантаження Stage 1 моделі...")
    model = load_model(MODEL_PATH)

    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers[-5:]:
        layer.trainable = True

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(LEARNING_RATE),
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    log("Тренування Stage 1...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"models/stage1_binary_{version}.keras")
    model.save(MODEL_PATH)

    acc  = history.history["val_accuracy"][-1]
    loss = history.history["val_loss"][-1]
    log(f"Stage 1 готово. Val accuracy: {acc:.4f} | Val loss: {loss:.4f}")

    shutil.rmtree(combined)

    return {
        "val_accuracy": round(acc, 4),
        "val_loss": round(loss, 4),
        "epochs": len(history.history["val_accuracy"]),
        "history_accuracy":     [round(x, 4) for x in history.history["accuracy"]],
        "history_val_accuracy": [round(x, 4) for x in history.history["val_accuracy"]],
        "history_loss":         [round(x, 4) for x in history.history["loss"]],
        "history_val_loss":     [round(x, 4) for x in history.history["val_loss"]],
    }

# ── Перенавчання Stage 2 (вид динозавра) ──────────────────────────────────────
def retrain_stage2_dino():
    log("=== Перенавчання Stage 2 (класифікація виду динозавра) ===")

    if not os.path.exists(TEMP_DINO_FOLDER):
        log("Немає нових зразків для Stage 2. Пропускаємо.")
        return None

    combined = "dataset/dino_combined"
    copy_images(DINO_FOLDER, combined, recursive=True)
    copy_images(TEMP_DINO_FOLDER, combined, recursive=True)

    datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        combined,
        target_size=IMG_SIZE_BIG,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        combined,
        target_size=IMG_SIZE_BIG,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    if train_gen.samples == 0:
        log("Немає зображень. Пропускаємо.")
        shutil.rmtree(combined)
        return None

    classes = list(train_gen.class_indices.keys())
    classes_dict = {str(i): name for i, name in enumerate(classes)}

    log("Завантаження Stage 2 моделі...")
    model = load_model(MODEL_PATH_DINO_CLASS)

    # Перевіряємо чи змінилась кількість класів
    with open(DINO_CLASSES_PATH, "r", encoding="utf-8") as f:
        current_classes = json.load(f)

    current_count = len(current_classes)
    new_count = len(classes)

    new_class_names = []

    if new_count != current_count:
        log(f"Кількість класів змінилась: {current_count} → {new_count}. Перебудовуємо вихідний шар...")
        from tensorflow.keras.layers import Dense
        from tensorflow.keras import Model

        # Знаходимо нові класи
        old_class_names = set(current_classes.values())
        new_class_names = [c for c in classes if c not in old_class_names]
        log(f"Нові класи: {new_class_names}")

        base_output = model.layers[-2].output
        new_output = Dense(new_count, activation="softmax", name="new_predictions")(base_output)
        model = Model(inputs=model.input, outputs=new_output)
        log(f"Архітектура оновлена: {current_count} → {new_count} класів")
    else:
        log(f"Кількість класів не змінилась ({current_count}). Продовжуємо fine-tuning.")

    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers[-5:]:
        layer.trainable = True

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(LEARNING_RATE),
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    log("Тренування Stage 2...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"models/stage2_dino_species_{version}.keras")
    model.save(MODEL_PATH_DINO_CLASS)

    # Зберігаємо оновлені класи після успішного навчання
    with open(DINO_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes_dict, f, ensure_ascii=False, indent=2)
    log(f"Класи оновлено: {classes}")

    acc  = history.history["val_accuracy"][-1]
    loss = history.history["val_loss"][-1]
    log(f"Stage 2 готово. Val accuracy: {acc:.4f} | Val loss: {loss:.4f}")

    shutil.rmtree(combined)

    return {
        "val_accuracy": round(acc, 4),
        "val_loss": round(loss, 4),
        "epochs": len(history.history["val_accuracy"]),
        "history_accuracy":     [round(x, 4) for x in history.history["accuracy"]],
        "history_val_accuracy": [round(x, 4) for x in history.history["val_accuracy"]],
        "history_loss":         [round(x, 4) for x in history.history["loss"]],
        "history_val_loss":     [round(x, 4) for x in history.history["val_loss"]],
        "new_classes": new_class_names,
    }

# ── Головна функція ────────────────────────────────────────────────────────────
def main(retrain_id: str = None):
    log("====== Запуск перенавчання DinoTerra ======")

    images = fetch_retrain_images()
    log(f"Отримано фото для перенавчання: {len(images)}")

    if len(images) < RETRAIN_THRESHOLD:
        log(f"Недостатньо фото ({len(images)}/{RETRAIN_THRESHOLD}). Виходимо.")
        # Повідомляємо що ретрен не відбувся
        if retrain_id:
            requests.post(
                f"{NODE_API_URL}/api/v1/ml/admin/retrain/done",
                json={
                    "retrainId": retrain_id,
                    "status": "FAILED",
                    "imagesUsed": len(images),
                }
            )
        return

    save_temp_images(images)

    stage1_result = retrain_stage1()
    stage2_result = retrain_stage2_dino()

    # Збираємо нові класи якщо були
    new_classes = stage2_result.get("new_classes", []) if stage2_result else []

    # Видаляємо тимчасові папки
    if os.path.exists("dataset/temp_retrain"):
        shutil.rmtree("dataset/temp_retrain")
        log("Тимчасові папки видалено")

    # Видаляємо фото з MongoDB
    try:
        response = requests.delete(f"{NODE_API_URL}/api/v1/ml/admin/retrain-images")
        log(f"Фото видалено з MongoDB: {response.json()}")
    except Exception as e:
        log(f"Помилка видалення фото з MongoDB: {e}")

    # Повідомляємо Node.js що ретрен завершено з результатами
    try:
        payload = {
            "retrainId": retrain_id,
            "imagesUsed": len(images),
            "newClasses": new_classes,
            "status": "DONE",
            "stage1": stage1_result,
            "stage2": stage2_result,
        }
        response = requests.post(
            f"{NODE_API_URL}/api/v1/ml/admin/retrain/done",
            json=payload,
        )
        log(f"Node.js повідомлено: {response.json()}")
    except Exception as e:
        log(f"Помилка повідомлення Node.js: {e}")

    log("====== Перенавчання завершено ======")


if __name__ == "__main__":
    import sys
    retrain_id = sys.argv[1] if len(sys.argv) > 1 else None
    main(retrain_id)