import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Ścieżka do katalogu z danymi
DATA_DIR = "C:/Users/Adrian/Desktop/KNO_repo/Lab07/photos"

# Parametry obrazków
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Wczytanie danych z katalogu
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None
)

dataset = dataset.map(lambda x: x / 255.0)  # Normalizacja obrazków

# Augmentacja danych
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

dataset = dataset.map(lambda x: (augmentation(x), x))

# Definicja autoenkodera
latent_dim = 48

# Encoder
encoder = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Flatten(),
    layers.Dense(latent_dim)
])

# Decoder
decoder = models.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(32 * 32 * 64, activation='relu'),
    layers.Reshape((32, 32, 64)),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
])

# Autoencoder
inputs = layers.Input(shape=(128, 128, 3))
encoded = encoder(inputs)
decoded = decoder(encoded)
autoencoder = models.Model(inputs, decoded)

# Kompilacja modelu
autoencoder.compile(optimizer='adam', loss='mse')

# Trening
autoencoder.fit(dataset, epochs=20)

# Zadanie dodatkowe: Transfer Learning
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

transfer_encoder = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(latent_dim)
])

# Nowy autoenkoder z transfer learning
transfer_inputs = layers.Input(shape=(128, 128, 3))
transfer_encoded = transfer_encoder(transfer_inputs)
transfer_decoded = decoder(transfer_encoded)
transfer_autoencoder = models.Model(transfer_inputs, transfer_decoded)

transfer_autoencoder.compile(optimizer='adam', loss='mse')

# Trening autoenkodera z transfer learning
transfer_autoencoder.fit(dataset, epochs=20)

# Generowanie obrazka z latent space
latent_vector = np.random.uniform(-1, 1, (1, 48))  # Losowe wartości w przestrzeni latentnej

  # Współrzędne w przestrzeni latentnej
generated_image = decoder.predict(latent_vector)

# Zapis wygenerowanego obrazka
import matplotlib.pyplot as plt
plt.imshow(generated_image[0])
plt.axis('off')
plt.savefig("generated_image.png")
