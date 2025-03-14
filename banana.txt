//Banana life span prediction

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Define the path
dataset_dir = "E:\\Downloads\\BANANA"

# Image data generators with augmentation for training data and rescaling for test data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Set the validation split ratio
)

# Training generator
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Set as training data
)

# Validation generator
val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Define a separate data generator for the test set (without data augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Function to split a portion of validation data as test data
def get_test_data(val_generator, num_samples):
    test_images = []
    test_labels = []
    for _ in range(num_samples):
        img, label = next(val_generator)
        test_images.append(img)
        test_labels.append(label)
    return np.vstack(test_images), np.vstack(test_labels)

# Assuming we want 20% of the validation data as test data
num_test_samples = int(0.2 * val_generator.samples)
test_images, test_labels = get_test_data(val_generator, num_test_samples)

# Load the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=val_generator
)

# Unfreeze some layers and fine-tune
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
    layer.trainable = True

# Recompile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=val_generator
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}')

# Save the trained model
model.save('banana_quality_model.keras')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_finetune.history['loss'])
plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

def predict_eatable_days(model, img_path):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    except FileNotFoundError as e:
        print(f"File not found: {img_path}")
        print("Available files in the directory:")
        print(os.listdir(os.path.dirname(img_path)))
        raise e

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    quality_rating = np.argmax(prediction) + 1  # Convert index to 1-5 rating

    # Mapping quality rating to eatable days
    eatable_days_mapping = {
        5: 0,
        4: 0.5,
        3: 1.5,
        2: 2.5,
        1: 6
    }

    eatable_days = eatable_days_mapping[quality_rating]

    return quality_rating, eatable_days

# Example usage
img_path = 'E:\\Downloads\\ripe-banana-close-crop-1325832.jpg'
quality_rating, eatable_days = predict_eatable_days(model, img_path)
print(f'Quality Rating: {quality_rating}, Eatable for: {eatable_days} days')


