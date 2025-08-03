import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration and Hyperparameters ---
TRAIN_DIR = "Face Mask Dataset/Train"
TEST_DIR = "Face Mask Dataset/Test"
VALIDATION_DIR = "Face Mask Dataset/Validation"

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10 # Start with 10, can increase for better accuracy
LEARNING_RATE = 1e-4

# --- Data Loading and Augmentation ---
print("[INFO] Loading and augmenting data...")

# Create data generators with augmentation for the training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# For validation and test sets, we only need to preprocess the input
validation_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# --- Model Building (Transfer Learning) ---
print("[INFO] Building model...")

# Load the MobileNetV2 network, ensuring the head FC layer sets are left off
base_model = MobileNetV2(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(len(train_generator.class_indices), activation="softmax")(head_model)

# Place the head FC model on top of the base model (this will become the actual model we train)
model = Model(inputs=base_model.input, outputs=head_model)

# Freeze all layers in the base model so they won't be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# --- Compile and Train the Model ---
print("[INFO] Compiling model...")
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("[INFO] Training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=EPOCHS
)

# --- Evaluate the Model ---
print("[INFO] Evaluating network...")
predictions = model.predict(test_generator, batch_size=BATCH_SIZE)
# For each image, find the index of the label with the corresponding largest predicted probability
predictions = np.argmax(predictions, axis=1)

print(classification_report(test_generator.classes, predictions,
                            target_names=test_generator.class_indices.keys()))

# --- Save the Model ---
print("[INFO] Saving mask detector model...")
model.save("mask_classifier.model", save_format="h5")

# --- Plot the Training History ---
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")