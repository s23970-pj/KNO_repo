import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


#wczytanie db
data_file_path = '/Users/adriangoik/Desktop/KNO_repo/Lab05/wine/wine.data'
column_names = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
    'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
    'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]
wine_data = pd.read_csv(data_file_path, header=None, names=column_names)

X = wine_data.drop(columns=['Class']) #podział na etykiety
y = wine_data['Class'] - 1  # Etykiety zaczynają się od 0

# Podział na zbiory treningowe i walidacyjne
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Skalowanie cech
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

#Definicja modelu w postaci klasy Keras
class CustomModel(Model):
    def __init__(self, num_units, dropout_rate):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(num_units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(3, activation='softmax')  # 3 klasy w danych wina

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        return self.dense2(x)

# Funkcja budująca model (używana przez Keras Tuner)
def build_model(hp):
    num_units = hp.Int('num_units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.3, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = CustomModel(num_units=num_units, dropout_rate=dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=2,
    directory='my_dir',
    project_name='wine_hyperparam_tuning'
)

#  Rozpoczęcie wyszukiwania z keras tuner
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16)

# Pobranie najlepszego modelu i hiperparametrów
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Najlepsze hiperparametry: num_units={best_hps.get('num_units')}, "
      f"dropout_rate={best_hps.get('dropout_rate')}, "
      f"learning_rate={best_hps.get('learning_rate')}")

#Trening modelu z najlepszymi hiperparametrami
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

#Ewaluacja modelu
loss, accuracy = best_model.evaluate(X_val, y_val)
print(f"Dokładność walidacyjna: {accuracy:.4f}")
