import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================
# Dataset Paths
# ======================
TRAIN_DIR = "lung_cancer_MRI_dataset/train"
VAL_DIR = "lung_cancer_MRI_dataset/validate"

# ======================
# Data Generators
# ======================
datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224),
    batch_size=16, class_mode="binary"
)

val_data = datagen.flow_from_directory(
    VAL_DIR, target_size=(224, 224),
    batch_size=16, class_mode="binary"
)

# ======================
# Transfer Learning Model
# ======================
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False   # Freeze imagenet layers first

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ======================
# Callbacks
# ======================
checkpoint = ModelCheckpoint(
    "lung_cancer_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ======================
# Training
# ======================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)

print("Training Completed.")
