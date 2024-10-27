import tensorflow as tf

def solve_linear_system(A, b):
    # Rozwiązanie równania Ax = b
    solution = tf.linalg.solve(A, b)
    return solution

# Przykład
A = tf.constant([[2.0, 1.0], [1.0, 3.0]], dtype=tf.float32)
b = tf.constant([[5.0], [6.0]], dtype=tf.float32)
solution = solve_linear_system(A, b)
print("Rozwiązanie układu równań:", solution.numpy())
