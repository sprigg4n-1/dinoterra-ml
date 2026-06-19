# ── Розміри зображень ─────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)         # Stage 1 (EfficientNetB0)
IMG_SIZE_BIG = (224, 224)    # ResNet50 (не-динозавр)
IMG_SIZE_STAGE2 = (300, 300) # Stage 2 — EfficientNetV2S

# ── Параметри навчання ────────────────────────────────────────────────────────
BATCH_SIZE = 64
INITIAL_LABELED_RATIO = 0.25
ACTIVE_LEARNING_STEP = 0.03
EPOCHS = 30
LEARNING_RATE = 1e-5

# ── Шляхи до датасету ─────────────────────────────────────────────────────────
DINO_FOLDER         = "dataset/dinosaur"
NON_DINO_FOLDER     = "dataset/not_dinosaur"

NEW_DINO_FOLDER     = "dataset/new_dino"       
NEW_NON_DINO_FOLDER = "dataset/new_non_dino"   

PENDING_FOLDER      = "dataset/pending"        
CONFIRMED_FOLDER    = "dataset/confirmed"      

# ── Базові моделі (користувач кладе сюди готові, ніколи не перезаписуються) ──
BASE_MODEL_PATH           = "base/stage1_efficientnetb0.keras"
BASE_MODEL_PATH_DINO_CLASS = "base/stage2_dino_species.keras"
BASE_DINO_CLASSES_PATH    = "base/stage2_dino_classes.json"

# ── Runtime-моделі (перезаписуються після кожного ретрену) ────────────────────
MODEL_PATH              = "models/stage1_binary.keras"
MODEL_PATH_DINO_CLASS   = "models/stage2_dino_species.keras"
MODEL_PATH_NON_DINO_CLASS = "models/stage2_non_dino_species.keras"

# ── Шляхи до класів ───────────────────────────────────────────────────────────
DINO_CLASSES_PATH       = "models/stage2_dino_classes.json"
NON_DINO_CLASSES_PATH   = "models/stage2_non_dino_classes.json"

# ── Налаштування перенавчання ─────────────────────────────────────────────────
RETRAIN_THRESHOLD = 10
STATS_PATH        = "models/stats.json"

# Кількість останніх шарів (не BatchNorm) які тренуються при fine-tuning.
# Решта шарів заморожена — модель не "забуває" те, що вже знає.
# Stage 1 (бінарна): зазвичай достатньо 5–10
# Stage 2 (класифікація виду): 10–20 якщо додаються нові класи
FINE_TUNE_LAYERS = 10