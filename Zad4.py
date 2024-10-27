import tensorflow as tf
import sys
import numpy as np

def solve_from_command_line(args):
    # Przetwarzanie argumentów
    data = list(map(float, args[1:]))  # pierwszy argument to nazwa pliku

    # Sprawdzanie liczby elementów
    if len(data) < 2:
        print("Za mało argumentów. Podaj macierz A oraz wektor b.")
        return

    # Obliczanie rozmiaru macierzy n x n
    num_elements = len(data) - 1
    n = int(np.sqrt(num_elements))

    # Sprawdzanie, czy liczba elementów jest zgodna z wymaganym formatem n x n + n
    if n * n + n != len(data):
        print("Nieprawidłowa liczba argumentów. Podaj macierz n x n oraz wektor o długości n.")
        return

    # Rozdzielenie na macierz A i wektor b
    A_data = data[:n * n]
    b_data = data[n * n:]

    # Konwersja do tensora
    A = tf.constant(np.reshape(A_data, (n, n)), dtype=tf.float32)
    b = tf.constant(np.reshape(b_data, (n, 1)), dtype=tf.float32)

    # Rozwiązywanie układu równań
    try:
        solution = tf.linalg.solve(A, b)
        print("Rozwiązanie układu równań:", solution.numpy())
    except tf.errors.InvalidArgumentError:
        print("Brak rozwiązania lub nieodpowiednie dane.")

# Wywołanie z linii komend: np. python program.py 2 1 1 3 5 6 (dla macierzy [[2,1],[1,3]] i wektora [5,6])
if __name__ == "__main__":
    solve_from_command_line(sys.argv)
