from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import itertools
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Ścieżka do pliku
file_path = Path('/Users/adriangoik/Desktop/KNO_repo/Lab04/best_model_hp.h5')
#Przygotowanie danych
# Wczytaj dane
column_names = [
    "Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
    "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
    "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"
]
wine_data = pd.read_csv('/Users/adriangoik/Desktop/KNO_repo/Lab04/wine/wine.data', header=None, names=column_names)

# Podział danych na cechy (X) i klasy (y)
X = wine_data.drop("Class", axis=1)
y = wine_data["Class"] - 1  # Klasy zmienione na 0, 1, 2 (dla TensorFlow)


# Funkcja do podziału danych na zbiory
def prepare_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)


# Funkcja tworząca model
def create_model(hp_units, hp_learning_rate, hp_dropout_rate, input_shape):
    """
    Tworzy model z podanymi hiperparametrami.
    """
    model = Sequential([
        Input(shape=input_shape),
        Dense(units=hp_units, activation='relu'),
        Dropout(hp_dropout_rate),
        Dense(units=hp_units // 2, activation='relu'),
        Dense(3, activation='softmax')  # 3 klasy w zbiorze danych
    ])
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Hiperparametry do optymalizacji
units_options = [64, 128]  # Liczba neuronów
learning_rate_options = [0.001, 0.01]  # Tempo uczenia
dropout_rate_options = [0.2, 0.3]  # Dropout

param_combinations = list(itertools.product(units_options, learning_rate_options, dropout_rate_options))

# Model bazowy (baseline)
baseline_model_path = '/Users/adriangoik/Desktop/KNO_repo/Lab03/best_model.h5'
baseline_model = load_model(baseline_model_path)

# Konwersja etykiet na one-hot encoded
y_val_one_hot = to_categorical(y_val, num_classes=3)
y_test_one_hot = to_categorical(y_test, num_classes=3)

# Ewaluacja modelu bazowego
baseline_val_loss, baseline_val_acc = baseline_model.evaluate(X_val, y_val_one_hot, verbose=0)
baseline_test_loss, baseline_test_acc = baseline_model.evaluate(X_test, y_test_one_hot, verbose=0)

print(f"Baseline validation accuracy: {baseline_val_acc}")
print(f"Baseline test accuracy: {baseline_test_acc}")

# Eksperymenty
results = []

for params in param_combinations:
    hp_units, hp_learning_rate, hp_dropout_rate = params
    print(f"Testing params: Units={hp_units}, Learning Rate={hp_learning_rate}, Dropout={hp_dropout_rate}")

    model = create_model(hp_units, hp_learning_rate, hp_dropout_rate, input_shape=(X_train.shape[1],))
    #tensorboard razem z timestampem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{timestamp}_units_{hp_units}_lr_{hp_learning_rate}_dropout_{hp_dropout_rate}"
    tensorboard = TensorBoard(log_dir=log_dir)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=0,
        callbacks=[tensorboard]
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    results.append({
        'params': params,
        'val_accuracy': val_acc
    })

# Wybór najlepszego modelu
best_model_params = max(results, key=lambda x: x['val_accuracy'])
print(f"Best params: {best_model_params['params']} with validation accuracy: {best_model_params['val_accuracy']}")

best_units, best_learning_rate, best_dropout_rate = best_model_params['params']
best_model = create_model(best_units, best_learning_rate, best_dropout_rate, input_shape=(X_train.shape[1],))
best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

# Ewaluacja na zbiorze testowym
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
print(f"Test accuracy: {test_acc}")
print(f"val accuracy: {val_acc}")

# Sprawdzenie, czy plik istnieje
if file_path.exists():
    print("Model already exists. Skipping save.")
else:
    best_model.save(file_path)
    print("Best model saved as 'best_model_hp.h5'")

# Tworzenie tabeli porównawczej
comparison_data = {
    "Model": ["Baseline", "Optimized"],
    "Validation Accuracy": [baseline_val_acc, val_acc],
    "Test Accuracy": [baseline_test_acc, test_acc],
    "Validation Loss": [baseline_val_loss, val_loss],
    "Test Loss": [baseline_test_loss, test_loss],
}

comparison_df = pd.DataFrame(comparison_data)

# Wyświetlenie porównania
print("\nPorównanie wyników modeli:")
print(comparison_df)

