import tensorflow as tf
import math

def rotate_point(x, y, angle_degrees):
    # Przeliczanie kąta na radiany i tworzenie tensora kąta
    angle_radians = tf.constant(math.radians(angle_degrees), dtype=tf.float32)
    # Macierz obrotu
    rotation_matrix = tf.stack([
        [tf.cos(angle_radians), -tf.sin(angle_radians)],
        [tf.sin(angle_radians), tf.cos(angle_radians)]
    ])
    # Punkt w formie tensora
    point = tf.constant([x, y], dtype=tf.float32)
    # Mnożenie macierzy przez wektor punktu
    rotated_point = tf.linalg.matvec(rotation_matrix, point)
    return rotated_point

# Przykład użycia
rotated = rotate_point(1.0, 0.0, 90)  # obrót o 90 stopni wokół punktu (0, 0)
print("Obrócony punkt:", rotated.numpy())


