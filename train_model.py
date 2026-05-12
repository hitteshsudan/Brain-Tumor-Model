import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# ==========================
# Config
# ==========================
TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/validation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 40
MODEL_PATH = "Brain_Tumor_Model.h5"
NUM_CLASSES = 4

# ==========================
# Data Generators
# ✅ Slightly more augmentation than your current version
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"✅ Class indices: {train_generator.class_indices}")

# ==========================
# Fix Class Imbalance
# ==========================
classes = np.unique(train_generator.classes)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_generator.classes
)
class_weight = dict(enumerate(weights))
print(f"✅ Class weights: {class_weight}")

# ==========================
# Build Model
# ✅ Added BatchNormalization after pooling
# ✅ Added extra Dense layer
# ✅ Added L2 regularization (weak 0.0001)
# ✅ Backbone fully frozen — no Phase 2
# ==========================
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),

    # ✅ BatchNorm stabilizes training
    BatchNormalization(),

    # ✅ Bigger first layer
    Dense(512, activation="relu", kernel_regularizer=l2(0.0001)),
    Dropout(0.5),

    # ✅ Same middle layer as your working version
    Dense(256, activation="relu", kernel_regularizer=l2(0.0001)),
    Dropout(0.4),

    # ✅ Extra layer for better 4-class separation
    Dense(128, activation="relu", kernel_regularizer=l2(0.0001)),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# Callbacks
# ==========================
callbacks = [
    EarlyStopping(
        patience=8,
        restore_best_weights=True,
        monitor="val_accuracy",
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ==========================
# Train — NO Phase 2 Ever
# ==========================
print("\n🚀 Training started...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight
)

model.save(MODEL_PATH)
best_acc = max(history.history['val_accuracy']) * 100
print(f"\n💾 Model saved to {MODEL_PATH}")
print(f"✅ Best val_accuracy: {best_acc:.2f}%")
print(f"✅ Label order: {list(train_generator.class_indices.keys())}")
