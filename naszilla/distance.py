import numpy as np

def add_noise(dist_matrix, eps):
    noise = np.random.normal(eps, eps/2, dist_matrix.shape)