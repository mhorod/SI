import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

TRAIN_DIR = 'dataset/fruits-360_dataset/fruits-360/Training'
TEST_DIR = 'dataset/fruits-360_dataset/fruits-360/Test'

IMG_SIZE = (100, 100)
BATCH_SIZE = 64
SHUFFLE = True
SEED = 42
VALIDATION_SPLIT = 0.2

# Loading training data
train_data = image_dataset_from_directory(
    TRAIN_DIR, 
    shuffle=SHUFFLE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='training', 
    image_size=IMG_SIZE,
    seed=SEED
)

# Load validation data
validation_data = image_dataset_from_directory(
    TRAIN_DIR, 
    shuffle=SHUFFLE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='validation', 
    image_size=IMG_SIZE,
    seed=SEED
)

# Load test data
test_data = image_dataset_from_directory(
    TEST_DIR, 
    shuffle=SHUFFLE, 
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

NUMBER_OF_CLASSES = 131