import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE

    random_vector = np.random.randn(data.shape[0])
    eigenvector = random_vector
    new_eigenvector = random_vector
    eigenvalue = 0

    for i in range(num_steps):
        eigenvector = new_eigenvector

        if eigenvector[0] < 0:
            eigenvector *= -1

        eigenvalue = eigenvector.T.dot(data.dot(eigenvector)) / eigenvector.T.dot(eigenvector)

        ax = data.dot(eigenvector)
        new_eigenvector = ax / np.linalg.norm(ax)

    return float(eigenvalue), eigenvector
