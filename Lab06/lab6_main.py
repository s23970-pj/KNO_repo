import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
import numpy as np

# Załadowanie danych MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizacja danych
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-hot encoding dla etykiet
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

'''Sieć warstwowa Fully connected dense layers dla porównania'''
# Przekształcenie danych do wektora (flatten)
train_images_flat = train_images.reshape((60000, 28 * 28))
test_images_flat = test_images.reshape((10000, 28 * 28))

# Budowa modelu FCNN
fcnn_model = models.Sequential()
fcnn_model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
fcnn_model.add(layers.Dense(256, activation='relu'))
fcnn_model.add(layers.Dense(10, activation='softmax'))

# Kompilacja modelu FCNN
fcnn_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Trenowanie modelu FCNN
print("\nTrening modelu FCNN...\n")
fcnn_model.fit(train_images_flat, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Ewaluacja modelu FCNN
fcnn_test_loss, fcnn_test_acc = fcnn_model.evaluate(test_images_flat, test_labels)
print(f'\nDokładność FCNN na zbiorze testowym: {fcnn_test_acc}')

'''Sieć z zadania 6'''
# Przekształcenie danych do formatu 28x28x1 (dla CNN)
train_images_cnn = train_images.reshape((60000, 28, 28, 1))
test_images_cnn = test_images.reshape((10000, 28, 28, 1))

# Budowa rozszerzonego modelu CNN
cnn_model = models.Sequential()

# Pierwsza warstwa konwolucyjna + pooling
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(layers.MaxPooling2D((2, 2)))

# Analiza po pierwszym bloku
print("Po pierwszym bloku konwolucyjnym:")
cnn_model.summary()

# Druga warstwa konwolucyjna + pooling
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))

# Analiza po drugim bloku
print("\nPo drugim bloku konwolucyjnym:")
cnn_model.summary()

# Trzecia warstwa konwolucyjna (bez pooling)
cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Analiza po trzeciej warstwie konwolucyjnej
print("\nPo trzeciej warstwie konwolucyjnej:")
cnn_model.summary()

# Flatten i warstwa w pełni połączona
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dense(10, activation='softmax'))

# Finalne podsumowanie modelu
print("\nFinalny model CNN:")
cnn_model.summary()

# Kompilacja modelu CNN
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# UWAGA: Teraz dopiero rozpoczynamy trening CNN
print("\nTrening modelu CNN:\n")
cnn_model.fit(train_images_cnn, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Ewaluacja modelu CNN
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_images_cnn, test_labels)
print(f'\nDokładność CNN na zbiorze testowym: {cnn_test_acc}')

'''Porównanie wyników'''

print("\n---- Podsumowanie ----")
print(f'Dokładność sieci w pełni połączonej (FCNN): {fcnn_test_acc * 100:.2f}%')
print(f'Dokładność sieci konwolucyjnej (CNN): {cnn_test_acc * 100:.2f}%')

# ------------------ Funkcja do klasyfikacji pojedynczego obrazu ------------------

def predict_digit(image_path):
    # Załaduj obraz i przekształć do rozmiaru 28x28 pikseli (skala szarości)
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))

    # Konwersja obrazu na tablicę NumPy
    img_array = img_to_array(img)

    # Normalizacja wartości pikseli (tak samo jak dla zbioru treningowego)
    img_array = img_array.astype('float32') / 255.0

    # Dodanie wymiaru batch_size (model oczekuje 4D: [batch_size, height, width, channels])
    img_array = np.expand_dims(img_array, axis=0)

    # Przewidywanie klasy przy użyciu modelu CNN
    prediction = cnn_model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    print(f'Przewidywana cyfra: {predicted_digit}')
    return predicted_digit

# Wywołanie funkcji na obrazie (upewnij się, że plik istnieje)
predict_digit('digit.png')
