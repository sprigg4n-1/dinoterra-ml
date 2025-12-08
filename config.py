# Розмір зображень
IMG_SIZE = (150, 150)

# Параметри навчання
BATCH_SIZE = 32
INITIAL_LABELED_RATIO = 0.3
ACTIVE_LEARNING_STEP = 0.02
EPOCHS = 5
LEARNING_RATE = 1e-4

# Шляхи до даних
DINO_FOLDER = "dataset/dinosaur"
NON_DINO_FOLDER = "dataset/not_dinosaur"

NEW_DINO_FOLDER = "dataset/new_dino"
NEW_NON_DINO_FOLDER = "dataset/new_non_dino"

# Шлях до збереженої моделі
MODEL_PATH = "models/cnn_model_active.h5"
