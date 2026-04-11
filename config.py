# Розмір зображень
IMG_SIZE = (150, 150)

# Параметри навчання
BATCH_SIZE = 64
INITIAL_LABELED_RATIO = 0.25
ACTIVE_LEARNING_STEP = 0.03
EPOCHS = 30
LEARNING_RATE = 1e-5

# Шляхи до даних
DINO_FOLDER = "dataset/dinosaur"
NON_DINO_FOLDER = "dataset/not_dinosaur"

NEW_DINO_FOLDER = "dataset/new_dino"
NEW_NON_DINO_FOLDER = "dataset/new_non_dino"

DINO_CLASSES_PATH = "models/stage2_dino_classes.json"

# Шлях до збереженої моделі
MODEL_PATH = "models/stage1_binary.keras"
MODEL_PATH_DINO_CLASS = "models/stage2_dino_species.keras"
MODEL_PATH_NON_DINO_CLASS = "models/stage2_non_dino_species.keras"


