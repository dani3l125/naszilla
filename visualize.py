import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from naszilla.coresets.k_means_coreset_via_robust_median import k_means_coreset_via_robust_median
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.ticker import PercentFormatter
import subprocess
from scipy import interpolate

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


def make_plots(acc_distances_to_representatives, estimated_points, labels, dataset, distance, zoom=0,
               function='single', k=100):
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1 = fig1.add_subplot(projection='3d')
    ax2 = fig2.add_subplot()
    distribution_list = []

    for i in np.unique(labels):
        points = estimated_points[labels == i, :]
        if points.shape[0] <= 3:
            continue
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        color = np.array([r, g, b])
        # hull = ConvexHull(points)
        if function == 'single':
            distance_to_representative = acc_distances_to_representatives[labels == i]
            ax1.set_zlabel('Accuracy distance from representative')
        elif function == 'mean':
            distance_to_representative = np.mean(acc_distances_to_representatives[labels == i])
            ax1.set_zlabel('average distance from representative in cluster')
        elif function == 'max':
            distance_to_representative = np.max(acc_distances_to_representatives[labels == i])
            ax1.set_zlabel('maximal distance from representative in cluster')
        ax1.scatter(estimated_points[labels == i, 0], estimated_points[labels == i, 1],
                    distance_to_representative * np.ones((np.sum(labels == i))),
                    s=5, cmap='viridis', color=color)
        distribution_list.append(distance_to_representative)
        try:
            hull = ConvexHull(points)
        except:
            continue
        ax2.scatter(estimated_points[labels == i, 0], estimated_points[labels == i, 1],
                    s=5, cmap='viridis', color=color)
        x_hull = np.append(points[hull.vertices, 0],
                           points[hull.vertices, 0][0])
        y_hull = np.append(points[hull.vertices, 1],
                           points[hull.vertices, 1][0])
        dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull],
                                        u=dist_along, s=0, per=1)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        # plot shape
        ax2.fill(interp_x, interp_y, '--', color=color, alpha=0.2)

    distribution_array = np.concatenate(distribution_list) if isinstance(distribution_list[0],
                                                                         np.ndarray) else np.array(distribution_list)

    fig3, (dax1, dax2) = plt.subplots(2, 1, sharex=True)
    fig3.subplots_adjust(hspace=0.2)  # adjust space between axes

    vsum = np.sum(np.histogram(distribution_array, bins=50)[0])
    # plot the same data on both axes
    dax1.yaxis.set_major_formatter(PercentFormatter(xmax=vsum))
    dax2.yaxis.set_major_formatter(PercentFormatter(xmax=vsum))
    dax1.hist(distribution_array, bins=50, density=False)
    dax2.hist(distribution_array, bins=50, density=False)
    # zoom-in / limit the view to different portions of the data
    dax1.set_ylim(bottom=.6 * vsum, top=vsum)
    dax2.set_ylim(bottom=0, top=.1 * vsum)
    # hide the spines between ax and ax2
    dax1.spines.bottom.set_visible(False)
    dax2.spines.top.set_visible(False)
    dax1.xaxis.tick_top()
    dax1.tick_params(labeltop=False)  # don't put tick labels at the top
    dax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    dax1.plot([0, 1], [0, 0], transform=dax1.transAxes, **kwargs)
    dax2.plot([0, 1], [1, 1], transform=dax2.transAxes, **kwargs)

    fig1.savefig(f'visualizations/{dataset}_{distance}_zoom{zoom}_{function}_k{k}_3d.png')
    fig2.savefig(f'visualizations/{dataset}_{distance}_zoom{zoom}_{function}_k{k}_2d.png')
    fig3.savefig(f'visualizations/{dataset}_{distance}_zoom{zoom}_{function}_k{k}_hist.png')
    fig1.clf()
    fig2.clf()
    fig3.clf()


if __name__ == '__main__':
    subprocess.run(['rm', '-r', 'visualizations'])
    subprocess.run(['mkdir', 'visualizations'])

    for k in [10, 25, 50, 100]:
        for dataset in ['cifar10', 'cifar100', 'imagenet']:
            for distance in ['nasbot', 'path', 'adj', 'lev']:
                indexes = np.random.randint(0, 15000, 1000)
                dist_matrix = np.load(f'distances/{distance}_dist.npy')[indexes][:, indexes]
                acc_dist_matrix = np.load(f'distances/{dataset}_dist.npy')[indexes][:, indexes]
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

                for function in ['single', 'max', 'mean']:
                    make_plots(acc_distances_to_representatives, estimated_points, labels, dataset, distance,
                               function=function, k=k)

                for zoom in [1, 2]:
                    sampled_clusters = np.random.randint(0, np.max(labels), 10)
                    sampled_points_indexes = \
                        np.concatenate([np.argwhere(labels == sampled_clusters[i]) for i in range(10)]).T[0]
                    dist_matrix = dist_matrix[sampled_points_indexes][:, sampled_points_indexes]
                    acc_dist_matrix = acc_dist_matrix[sampled_points_indexes][:, sampled_points_indexes]
                    estimated_points = estimated_points[sampled_points_indexes]
                    _, _, _, _, coreset_indexes = k_means_coreset_via_robust_median(estimated_points, k=k,
                                                                                    coreset_iteration_sample_size=1,
                                                                                    median_sample_size=200,
                                                                                    r=1,
                                                                                    sum_to_max=True)
                    points2coreset_dist_mat = distance_matrix(estimated_points, estimated_points[coreset_indexes])
                    labels = np.argmin(points2coreset_dist_mat, axis=1)
                    acc_distances_to_representatives = np.min(acc_dist_matrix[:, coreset_indexes], axis=1)

                    for function in ['single', 'max', 'mean']:
                        make_plots(acc_distances_to_representatives, estimated_points, labels, dataset, distance, zoom,
                                   function=function, k=k)
