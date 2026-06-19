import os
import json
import base64
import shutil
import requests
from collections import defaultdict
import numpy as np
from datetime import datetime
from PIL import Image as PILImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    IMG_SIZE, IMG_SIZE_BIG, IMG_SIZE_STAGE2,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    BASE_MODEL_PATH, BASE_MODEL_PATH_DINO_CLASS, BASE_DINO_CLASSES_PATH,
    MODEL_PATH, MODEL_PATH_DINO_CLASS,
    DINO_FOLDER, NON_DINO_FOLDER,
    DINO_CLASSES_PATH,
    RETRAIN_THRESHOLD,
    FINE_TUNE_LAYERS,
)

def _resolve(runtime_path: str, base_path: str) -> str:
    """Повертає runtime-модель якщо вже є перенавчена, інакше базову."""
    return runtime_path if os.path.exists(runtime_path) else base_path

NODE_API_URL = "http://localhost:9000"
ML_API_URL   = "http://localhost:8000"

TEMP_DINO_FOLDER     = "dataset/temp_retrain/dino"
TEMP_BINARY_DINO     = "dataset/temp_retrain/binary/dinosaur"
TEMP_BINARY_NON_DINO = "dataset/temp_retrain/binary/not_dinosaur"

MODEL_PATH_NEW      = "models/stage1_binary_new.keras"
MODEL_PATH_DINO_NEW = "models/stage2_dino_species_new.keras"

# ── Логування ──────────────────────────────────────────────────────────────────
def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ── Заморожування шарів для fine-tuning ───────────────────────────────────────
def freeze_for_finetuning(model, n_trainable: int = FINE_TUNE_LAYERS):
    """
    Заморожує всі шари моделі, крім останніх n_trainable (не рахуючи BatchNorm).

    BatchNormalization шари завжди залишаються замороженими — інакше вони
    оновлюють mean/variance і руйнують те, що модель вже вивчила (catastrophic forgetting).

    Args:
        model:        Keras-модель
        n_trainable:  кількість останніх шарів (без BatchNorm), що тренуються
    """
    # 1. Заморожуємо абсолютно все
    for layer in model.layers:
        layer.trainable = False

    # 2. Вибираємо останні n_trainable шарів, пропускаючи BatchNorm
    non_bn_layers = [l for l in model.layers if not isinstance(l, BatchNormalization)]
    layers_to_unfreeze = non_bn_layers[-n_trainable:]

    for layer in layers_to_unfreeze:
        layer.trainable = True

    # 3. Логуємо підсумок
    total      = len(model.layers)
    trainable  = sum(1 for l in model.layers if l.trainable)
    frozen     = total - trainable
    log(f"Заморожено шарів: {frozen}/{total} | Тренуються: {trainable} (без BatchNorm)")

    # Детальний вивід останніх шарів
    log("Стан останніх шарів:")
    for layer in model.layers[-max(n_trainable + 5, 10):]:
        status  = "✓ TRAIN " if layer.trainable else "✗ frozen"
        bn_note = " [BatchNorm — завжди frozen]" if isinstance(layer, BatchNormalization) else ""
        log(f"  [{status}] {layer.name:40s} ({type(layer).__name__}){bn_note}")

    return model

# ── Отримати базовий клас з назви папки ───────────────────────────────────────
def get_base_class(folder_name: str) -> str:
    return folder_name.split("_")[0]

# ── Автодетекція правильного порядку класів ────────────────────────────────────
def detect_class_order(model) -> list:
    """
    Перевіряє середній score на 20 дино-зображеннях.

    Якщо avg > 0.5:  модель навчена з dinosaur=1  → ["not_dinosaur", "dinosaur"]
    Якщо avg < 0.5:  модель навчена з dinosaur=0  → ["dinosaur", "not_dinosaur"]

    Без цього кроку навчання може боротись саме проти себе:
    loss=2.6, val_acc=0.25 — симптом інверсії класів.
    """
    scores = []
    count = 0
    for root, _, files in os.walk(DINO_FOLDER):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img = load_img(os.path.join(root, fname), target_size=IMG_SIZE)
                    arr = efficientnet_preprocess(np.expand_dims(img_to_array(img), axis=0))
                    score = float(model.predict(arr, verbose=0)[0][0])
                    scores.append(score)
                    count += 1
                except Exception:
                    pass
            if count >= 20:
                break
        if count >= 20:
            break

    avg = float(np.mean(scores)) if scores else 0.5
    log(f"Автодетекція класів: середній score на {count} дино-фото = {avg:.4f}")

    if avg > 0.5:
        log("  → dinosaur = клас 1  (classes=['not_dinosaur', 'dinosaur'])")
        return ["not_dinosaur", "dinosaur"]
    else:
        log("  → dinosaur = клас 0  (classes=['dinosaur', 'not_dinosaur'])")
        return ["dinosaur", "not_dinosaur"]

# ── Фільтрація зламаних фото ───────────────────────────────────────────────────
def clean_broken_images(folder: str):
    removed = 0
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, fname)
                try:
                    with PILImage.open(path) as img:
                        img.verify()
                except Exception:
                    os.remove(path)
                    removed += 1
    log(f"Видалено зламаних фото: {removed}")

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

# ── Копіювання зображень (звичайне) ───────────────────────────────────────────
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


# ── Перевірка моделі ───────────────────────────────────────────────────────────
def check_model(model, expected_classes: int):
    img_test = np.zeros((1, 224, 224, 3))
    preds = model.predict(img_test, verbose=0)
    actual = preds.shape[1]
    log(f"Перевірка моделі: очікується {expected_classes} класів, є {actual}")
    return actual == expected_classes

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

    clean_broken_images(combined)

    # ── Завантажуємо модель ДО генераторів, щоб визначити порядок класів ──────
    load_from = (MODEL_PATH_NEW if os.path.exists(MODEL_PATH_NEW)
                 else _resolve(MODEL_PATH, BASE_MODEL_PATH))
    log(f"Завантаження Stage 1 моделі: {load_from}")
    log(f"Розмір файлу: {os.path.getsize(load_from) / 1024 / 1024:.1f} MB")
    model = load_model(load_from)

    # ── Визначаємо правильний порядок класів (як у test_retrain.py) ───────────
    # flow_from_directory присвоює класи алфавітно, але модель могла бути
    # навчена з іншим порядком → без цього labels інвертовані → val_acc=0.25
    class_order = detect_class_order(model)

    datagen = ImageDataGenerator(
        preprocessing_function=efficientnet_preprocess,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        combined,
        classes=class_order,          # ← явний порядок, щоб збігався з моделлю
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    if train_gen.samples == 0:
        log("Немає зображень. Пропускаємо.")
        shutil.rmtree(combined)
        return None

    val_gen = datagen.flow_from_directory(
        combined,
        classes=class_order,          # ← той самий порядок для val
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    log(f"Клас-маппінг: {train_gen.class_indices}")
    log(f"Train: {train_gen.samples} | Val: {val_gen.samples}")

    # ── Заморожуємо всі шари крім останніх FINE_TUNE_LAYERS ──────────────────
    model = freeze_for_finetuning(model, n_trainable=FINE_TUNE_LAYERS)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(LEARNING_RATE),
        metrics=["accuracy"]
    )

    # ── Baseline перед навчанням (діагностика інверсії класів) ───────────────
    log("Baseline (до навчання):")
    baseline_loss, baseline_acc = model.evaluate(val_gen, verbose=0)
    log(f"  val_loss={baseline_loss:.4f} | val_accuracy={baseline_acc:.4f}")
    log(f"  {'OK — модель стартує зі знань' if baseline_acc > 0.75 else 'УВАГА — низька стартова точність, перевір клас-маппінг!'}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
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
    model.save(MODEL_PATH_NEW)
    log(f"Stage 1 збережено: models/stage1_binary_{version}.keras")

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

    # Групуємо папки з обох джерел за базовим класом (перше слово до "_")
    all_grouped = defaultdict(list)
    for source_dir in (DINO_FOLDER, TEMP_DINO_FOLDER):
        if not os.path.exists(source_dir):
            continue
        for sub in os.listdir(source_dir):
            full_path = os.path.join(source_dir, sub)
            if os.path.isdir(full_path):
                all_grouped[get_base_class(sub)].append(full_path)

    log(f"Визначені класи ({len(all_grouped)}):")
    for cls, paths in sorted(all_grouped.items()):
        log(f"  {cls}: {len(paths)} папки")

    # Копіюємо в combined — одна папка на клас
    for base_class, source_paths in all_grouped.items():
        target_dir = os.path.join(combined, base_class)
        os.makedirs(target_dir, exist_ok=True)
        for src_dir in source_paths:
            for fname in os.listdir(src_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                src_file = os.path.join(src_dir, fname)
                dst_file = os.path.join(target_dir, fname)
                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(fname)
                    dst_file = os.path.join(target_dir, f"{name}_{os.path.basename(src_dir)}{ext}")
                shutil.copy2(src_file, dst_file)

    clean_broken_images(combined)

    datagen = ImageDataGenerator(
        preprocessing_function=efficientnetv2_preprocess,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        combined,
        target_size=IMG_SIZE_STAGE2,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    if train_gen.samples == 0:
        log("Немає зображень. Пропускаємо.")
        shutil.rmtree(combined)
        return None

    val_gen = datagen.flow_from_directory(
        combined,
        target_size=IMG_SIZE_STAGE2,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    classes = list(train_gen.class_indices.keys())
    classes_dict = {str(i): name for i, name in enumerate(classes)}
    log(f"Класів для навчання: {len(classes)}")

    load_from = (MODEL_PATH_DINO_NEW if os.path.exists(MODEL_PATH_DINO_NEW)
                 else _resolve(MODEL_PATH_DINO_CLASS, BASE_MODEL_PATH_DINO_CLASS))
    log(f"Завантаження Stage 2 моделі: {load_from}")
    log(f"Розмір файлу: {os.path.getsize(load_from) / 1024 / 1024:.1f} MB")
    model = load_model(load_from)

    with open(BASE_DINO_CLASSES_PATH, "r", encoding="utf-8") as f:
        current_classes = json.load(f)

    known_class_names = set(current_classes.values())
    current_count = len(current_classes)
    new_count = len(classes)
    new_class_names = [c for c in classes if c not in known_class_names]

    log(f"Поточна модель: {current_count} класів | Новий датасет: {new_count} класів")

    if new_count != current_count:
        from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
        from tensorflow.keras import Model
        log(f"Нові класи: {new_class_names}. Перебудовуємо голову ({current_count} → {new_count})...")
        # EfficientNetV2S: layers[-2]=GAP, layers[-1]=head_model (вкладена підмодель)
        gap_output = model.layers[-2].output
        x = BatchNormalization()(gap_output)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.3)(x)
        new_output = Dense(new_count, activation="softmax", name="new_predictions")(x)
        model = Model(inputs=model.input, outputs=new_output)
        log(f"Голову перебудовано: {current_count} → {new_count} класів")
    else:
        log(f"Fine-tuning існуючої моделі ({current_count} класів)")

    # ── Заморожуємо всі шари крім останніх FINE_TUNE_LAYERS ──────────────────
    # При додаванні нових класів (rebuild вихідного шару) — можна збільшити
    n_train = FINE_TUNE_LAYERS * 2 if new_class_names else FINE_TUNE_LAYERS
    model = freeze_for_finetuning(model, n_trainable=n_train)

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
    model.save(MODEL_PATH_DINO_NEW)
    log(f"Stage 2 збережено: models/stage2_dino_species_{version}.keras")

    with open(DINO_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes_dict, f, ensure_ascii=False, indent=2)
    log(f"Класи оновлено: {len(classes)} класів")

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

    new_classes = stage2_result.get("new_classes", []) if stage2_result else []

    if os.path.exists("dataset/temp_retrain"):
        shutil.rmtree("dataset/temp_retrain")
        log("Тимчасові папки видалено")

    try:
        response = requests.delete(f"{NODE_API_URL}/api/v1/ml/admin/retrain-images")
        log(f"Фото видалено з MongoDB: {response.json()}")
    except Exception as e:
        log(f"Помилка видалення фото з MongoDB: {e}")

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

    try:
        requests.post(f"{ML_API_URL}/retrain_done")
        log("FastAPI повідомлено про hot-swap моделей")
    except Exception as e:
        log(f"Помилка повідомлення FastAPI: {e}")

    log("====== Перенавчання завершено ======")


if __name__ == "__main__":
    import sys
    retrain_id = sys.argv[1] if len(sys.argv) > 1 else None
    main(retrain_id)