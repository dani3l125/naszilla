import numpy as np
from scipy.spatial import distance_matrix

def get_points(dist_matrix, dim):
    '''
    Returns a matrix where each row is a point approximation
    '''
    m_row = np.tile(dist_matrix[0] ** 2, (dist_matrix.shape[0], 1))
    m_col = np.tile(dist_matrix.T[0] ** 2, (dist_matrix.shape[0], 1)).T
    M = (dist_matrix ** 2 + m_row + m_col) / 2

    w ,v = np.linalg.eigh(M.astype(np.float64))
    sign = np.tile(np.sign(w), (w.shape[0], 1))
    w_sqrt = np.tile(np.sqrt(np.abs(w.real)), (w.shape[0], 1))
    X = v.real * w_sqrt * sign

    ind = np.argpartition(np.abs(w), -dim)[-dim:]

    points = X.T[ind].T
    return points


n_points = 1500
original_points = np.random.rand(n_points, 3)
original_dist_matrix = distance_matrix(original_points, original_points)
for dim in range(n_points):
    estimated_points = get_points(original_dist_matrix, dim)
    estimated_dist_matrix = distance_matrix(estimated_points, estimated_points)
    print(f'Dimention = {dim+1}, Error = {np.sum(np.abs(estimated_dist_matrix - original_dist_matrix))}')

