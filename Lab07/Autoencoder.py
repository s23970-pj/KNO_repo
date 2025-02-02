import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Użycie backendu 'Agg' dla zapisu obrazów

# Ścieżka do katalogu z danymi
DATA_DIR = "/Users/adriangoik/Desktop/KNO_repo/Lab07/photos"

# Parametry obrazków
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# wczytywanie danych
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
    layers.RandomZoom(0.1), #poszerzenie datasetu o poprzerabiane obrazy
])

dataset = dataset.map(lambda x: (augmentation(x), x)) #doac dane z augumentacji żeby rzeczywiście poszerzyło bazę

# przestrzeń latentna - length of my sequences https://blog.keras.io/building-autoencoders-in-keras.html
latent_dim = 2 #rozmiar skompresowanej reprezentacji. u mnie mały więc ciężej złapać jakieś konkrety

# Encoder
encoder = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Flatten(),
    layers.Dense(latent_dim)
])

# Decoder - dodałem więcej warstw konwolucyjnych próbująć uzyskać lepszy obrazek
decoder = models.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(16 * 16 * 128, activation='relu'),
    layers.Reshape((16, 16, 128)),
    layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same'),
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
autoencoder.fit(dataset, epochs=50)

# Generowanie obrazka z latent space (autoencoder bez transfer learning)
latent_vector = np.array([[0.5, -0.5]])  # Losowe wartości w przestrzeni latentnej
generated_image = decoder.predict(latent_vector) #latent vector przetransformowany do obrazu

# Zapis obrazka wygenerowanego przez autoenkoder
import matplotlib.pyplot as plt
plt.imshow(generated_image[0])
plt.axis('off')
plt.title("Generated Image - Autoencoder")
plt.savefig("generated_image_autoencoder.png")  # Zapis obrazka
plt.show()

# Zadanie dodatkowe: Transfer Learning
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False #Zamrożenie wag mobileneta żeby zachować właściwości uczenia

transfer_encoder = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), #zmniejszenie przestrzeni do pojedynczego wektora
    layers.Dense(latent_dim) #kompresja do przestrzeni 2D
])

# Nowy autoenkoder z transfer learning
transfer_inputs = layers.Input(shape=(128, 128, 3))
transfer_encoded = transfer_encoder(transfer_inputs)
transfer_decoded = decoder(transfer_encoded)
transfer_autoencoder = models.Model(transfer_inputs, transfer_decoded)

transfer_autoencoder.compile(optimizer='adam', loss='mse')

# Trening autoenkodera z transfer learning
transfer_autoencoder.fit(dataset, epochs=50)

# Generowanie obrazka z latent space (autoencoder z transfer learning)
transfer_latent_vector = np.array([[0.5, -0.5]])  # Losowe wartości w przestrzeni latentnej
transfer_generated_image = decoder.predict(transfer_latent_vector)

# Zapis obrazka wygenerowanego przez autoenkoder z transfer learning
plt.imshow(transfer_generated_image[0])
plt.axis('off')
plt.title("Generated Image - Transfer Learning")
plt.savefig("generated_image_transfer_learning.png")
plt.show()
