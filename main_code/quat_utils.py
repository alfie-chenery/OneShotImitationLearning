import numpy as np

def quaternion_from_matrix(matrix):
    """
    Calculate quaternion from 3x3 rotation matrix.
    :param matrix: 3x3 rotation matrix
    :return: quaternion [w, x, y, z]
    """
    # Ensure the matrix is a valid rotation matrix
    assert is_rotation_matrix(matrix)

    # Extract rotation matrix elements for easier readability
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    # Calculate quaternion components
    qw = np.sqrt(1.0 + m00 + m11 + m22) / 2.0
    epsilon = 1e-6
    if qw < epsilon:
        # If qw is close to zero, handle it separately to avoid division by zero
        denom = 2.0 * np.sqrt(1 + m00 - m11 - m22)
        qx = (m10 + m01) / denom
        qy = 0.25 * denom
        qz = (m20 + m02) / denom
    else:
        # Normal calculation
        qx = (m21 - m12) / (4.0 * qw)
        qy = (m02 - m20) / (4.0 * qw)
        qz = (m10 - m01) / (4.0 * qw)

    return [qw, qx, qy, qz]

def is_rotation_matrix(matrix):
    """
    Check if a matrix is a valid rotation matrix.
    :param matrix: 3x3 matrix
    :return: True if the matrix is a rotation matrix, False otherwise
    """
    # Check if the matrix is square
    if not np.allclose(matrix.shape, (3, 3)):
        return False

    # Check if the matrix is orthonormal
    transposed = np.transpose(matrix)
    identity = np.dot(transposed, matrix)
    return np.allclose(identity, np.eye(3))

if __name__ == "__main__":
    # Example usage:
    # Assuming R is a 3x3 rotation matrix
    R = np.array([[1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]])

    # Convert rotation matrix to quaternion
    quaternion = quaternion_from_matrix(R)
    print("Quaternion:", quaternion)