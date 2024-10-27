import tensorflow as tf
import math

def rotate_tensor(point, angle_degrees):
    # Przeliczanie kąta na radiany i tworzenie tensora kąta
    angle_radians = tf.constant(math.radians(angle_degrees), dtype=tf.float32)
    # Tworzenie macierzy obrotu przy użyciu tf.stack
    rotation_matrix = tf.stack([
        [tf.cos(angle_radians), -tf.sin(angle_radians)],
        [tf.sin(angle_radians), tf.cos(angle_radians)]
    ])
    # Mnożenie macierzy przez wektor punktu
    rotated_point = tf.linalg.matvec(rotation_matrix, point)
    return rotated_point

# Przykład użycia
point = tf.constant([3.0, 4.0], dtype=tf.float32)
rotated = rotate_tensor(point, 45)  # obrót o 45 stopni
print("Obrócony punkt:", rotated.numpy())

