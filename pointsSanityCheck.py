import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from naszilla.coresets.k_means_coreset_via_robust_median import k_means_coreset_via_robust_median
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial import ConvexHull


def get_points(dist_matrix, dim):
    '''
    Returns a matrix where each row is a point approximation
    '''
    m_row = np.tile(dist_matrix[0] ** 2, (dist_matrix.shape[0], 1))
    m_col = np.tile(dist_matrix.T[0] ** 2, (dist_matrix.shape[0], 1)).T
    M = (dist_matrix ** 2 + m_row + m_col) / 2

    w, v = np.linalg.eigh(M.astype(np.float64))
    sign = np.tile(np.sign(w), (w.shape[0], 1))
    w_sqrt = np.tile(np.sqrt(np.abs(w.real)), (w.shape[0], 1))
    X = v.real * w_sqrt * sign

    ind = np.argpartition(np.abs(w), -dim)[-dim:]

    points = X.T[ind].T
    return points


k = 20

if __name__ == '__main__':
    # n_points = 1500
    # original_points = np.random.rand(n_points, 3)
    # original_dist_matrix = distance_matrix(original_points, original_points)
    indexes = np.random.randint(0, 15000, 1000)
    dist_matrix = np.load('distances/nasbot_dist.npy')[indexes][:, indexes]
    acc_dist_matrix = np.load('distances/cifar10_dist.npy')[indexes][:, indexes]
    P = np.zeros_like(dist_matrix)[:, :2]
    estimated_points = get_points(dist_matrix, 20)
    pca = PCA(n_components=2, svd_solver='full')
    estimated_points = pca.fit_transform(estimated_points)
    _, _, _, _, coreset_indexes = k_means_coreset_via_robust_median(estimated_points, k=k,
                                                                    coreset_iteration_sample_size=1,
                                                                    median_sample_size=200,
                                                                    r=1,
                                                                    sum_to_max=True)

    # points2coreset_dist_mat = dist_matrix[:, coreset_indexes]
    points2coreset_dist_mat = distance_matrix(estimated_points, estimated_points[coreset_indexes])
    labels = np.argmin(points2coreset_dist_mat, axis=1)
    acc_distances_to_representatives = np.min(acc_dist_matrix[:, coreset_indexes], axis=1)

    fig = plt.figure()
    ax1 = plt.subplot(111, projection='3d')
    ax2 = plt.subplot()
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    sns.displot(acc_distances_to_representatives)
    for i in np.unique(labels):
        points = estimated_points[labels == i, :]
        if points.shape[0] < 3:
            continue
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        color = np.array([r, g, b])
        hull = ConvexHull(points)
        distance_to_representative = np.max(acc_distances_to_representatives[labels == i])
        ax1.scatter(estimated_points[labels == i, 0], estimated_points[labels == i, 1],
                    distance_to_representative * np.ones((np.sum(labels == i))),
                    label="{:.2f}".format(distance_to_representative),
                    s=5, cmap='viridis', c=color)
        ax2.scatter(estimated_points[labels == i, 0], estimated_points[labels == i, 1],
                    label="{:.2f}".format(distance_to_representative),
                    s=5, cmap='viridis', c=color)
        x_hull = np.append(points[hull.vertices, 0],
                           points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1],
                           points[hull.vertices, 1][0])
        ax2.fill(x_hull, y_hull, alpha=0.3, c=color)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # plt.scatter(estimated_points[:,0], estimated_points[:,1])
    # plt.show()
