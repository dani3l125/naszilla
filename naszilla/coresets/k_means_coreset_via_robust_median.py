# matplotlib inline
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time
from sklearn.cluster import KMeans
import pandas as pd
import os
from scipy.spatial import distance_matrix
from random import randrange

global statistics_dict
statistics_dict = {'count': 0, 'mean': 1,
                   'std': 2, 'min': 3,
                   '25': 4, 'median': 5,
                   '75': 6, 'max': 7}


def flatten_list(input_list):
    flat_list = [item for sublist in input_list for item in sublist]
    return flat_list


def statistics_map(symbol):
    global statistics_dict
    return statistics_dict[symbol]


def compute_kmeans(Q, k, weights=None, n_init=30, max_iter=400):
    kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter).fit(Q, sample_weight=weights)
    return kmeans.cluster_centers_


def k_means_cost(Q, centers, weights=None):
    distance_table = euclidean_distances(Q, centers)
    distance_table.sort(axis=1)
    # print(distance_table)
    distances_to_sum = distance_table[:, 0] ** 2

    if weights is not None:
        cost = np.dot(weights, distances_to_sum)
    else:
        cost = np.sum(distances_to_sum)

    return cost


def knas_coreset(P, dist_matrix, **kwargs):
    if not kwargs['greedy']:
        _, _, coreset, _, coreset_indexes = k_means_coreset_via_robust_median(P, dist_matrix, **kwargs)
    else:
        coreset_indexes = k_centers_coreset_greedy(P, dist_matrix, **kwargs)
    # coreset_indexes = np.zeros(coreset.shape[0])
    # for i, point in enumerate(coreset):
    #     coreset_indexes[i] = np.where((P == point).all(axis=1))[0][0]

    if dist_matrix is None:
        points2coreset_dist_mat = distance_matrix(P, coreset)
    else:
        points2coreset_dist_mat = dist_matrix[:, coreset_indexes]
    labels = np.argmin(points2coreset_dist_mat, axis=1)

    return coreset_indexes.astype(int), labels.astype(int)

def k_centers_coreset_greedy(P,
                             dist_matrix=None,
                             coreset_iteration_sample_size=None,
                             k=None,
                             k_ratio=0,
                             sum_to_max=False,
                             median_sample_size=10,
                             tau_for_the_sampled_set=None,
                             tau_for_the_original_set=None,
                             Replace_in_coreset_sample=False,
                             use_threshold_method=False,
                             random_generation=False,
                             r=2):
    if dist_matrix in None:
        raise Exception('Greedy with geometric does not implemented')
    k_centers = [randrange(dist_matrix.shape[0])]

    for i in range(k-1):
        k_centers_array = np.array(k_centers)
        dist_matrix_from_centers = dist_matrix[k_centers_array]
        dist_matrix_from_centers[:, k_centers_array] = 10000
        min_idx_per_center = np.argmin(dist_matrix_from_centers, axis=1)
        min_value_per_center = np.min(dist_matrix_from_centers, axis=1)
        k_centers.append(min_idx_per_center[np.argmax(min_value_per_center)])

    return np.array(k_centers)


def k_means_coreset_via_robust_median(P,
                                      dist_matrix=None,
                                      greedy=False,
                                      coreset_iteration_sample_size=None,
                                      k=None,
                                      k_ratio=0,
                                      sum_to_max=False,
                                      median_sample_size=10,
                                      tau_for_the_sampled_set=None,
                                      tau_for_the_original_set=None,
                                      Replace_in_coreset_sample=False,
                                      use_threshold_method=False,
                                      random_generation=False,
                                      r=2):
    if median_sample_size < 1:
        median_sample_size = int(median_sample_size * P.shape[0])

    orig_indexes = np.arange(P.shape[0])

    delta_const = 10
    coreset_list = []
    coreset_index_list = []
    weights_list = []
    eps = 0.1
    if tau_for_the_sampled_set is None: tau_for_the_sampled_set = 15 / (16 * k)
    if tau_for_the_original_set is None: tau_for_the_original_set = 1 / (2 * k)
    if coreset_iteration_sample_size is None: coreset_iteration_sample_size = (k ** 2) / (eps ** 2)

    while tau_for_the_original_set * P.shape[0] > coreset_iteration_sample_size:
        idxes = np.random.choice(P.shape[0], np.min([P.shape[0], median_sample_size]), replace=False)
        sample_for_median = P[idxes]

        sample_for_median_dist_matrix = None if dist_matrix is None else dist_matrix[idxes][:, idxes]
        if random_generation:
            robust_median_q_idxes = np.random.choice(P.shape[0], 1)
            robust_median_q = P[robust_median_q_idxes].flatten()
            if not dist_matrix is None:
                opt_devided_by_size = np.sum(
                    ((euclidean_distances(robust_median_q.reshape(1, -1), sample_for_median)).flatten()) ** r) / (
                                              sample_for_median.shape[0] * delta_const)
            else:
                opt_devided_by_size = np.sum(
                    ((euclidean_distances(robust_median_q.reshape(1, -1), sample_for_median)).flatten()) ** r) / (
                                              sample_for_median.shape[0] * delta_const)
            # print(opt_devided_by_size)
        else:
            robust_median_q, opt_devided_by_size, robust_median_q_idx = compute_robust_median(sample_for_median,
                                                                                              tau_for_the_sampled_set,
                                                                                              use_threshold_method, r,
                                                                                              sample_for_median_dist_matrix,
                                                                                              is_max=sum_to_max)  # np.mean(P,axis=0)
            # print(opt_devided_by_size)
        if not use_threshold_method:
            sorted_indexes_of_distances_from_q = euclidean_distances(robust_median_q.reshape(1, -1), P).argsort()[0] \
                if dist_matrix is None else dist_matrix[robust_median_q_idx].argsort()  # TODO [0]
            closest_points_to_q = sorted_indexes_of_distances_from_q[:int(tau_for_the_original_set * P.shape[0])]
            far_points_from_q = sorted_indexes_of_distances_from_q[int(tau_for_the_original_set * P.shape[0]):]
        else:
            distances_matrix = (euclidean_distances(robust_median_q.reshape(1, -1), P) ** r)[0] \
                if dist_matrix is None else (dist_matrix[robust_median_q_idx] ** r)  # TODO [0]
            closest_points_to_q = np.argwhere(distances_matrix <= opt_devided_by_size).flatten()
            far_points_from_q = np.argwhere(distances_matrix > opt_devided_by_size).flatten()
        current_sample_size = coreset_iteration_sample_size
        if closest_points_to_q.shape[0] < current_sample_size:
            current_sample_size = closest_points_to_q.shape[0]
        idxes_for_coreset = np.random.choice(closest_points_to_q.shape[0],
                                             current_sample_size,
                                             replace=Replace_in_coreset_sample)
        # print(f'{P.shape}\n\n{closest_points_to_q.max()}\n\n{idxes_for_coreset.max()}')
        sampled_close_points = P[closest_points_to_q[idxes_for_coreset]]
        coreset_list.append(sampled_close_points)
        coreset_index_list.append(orig_indexes[closest_points_to_q[idxes_for_coreset]])

        weights_list.append(np.ones(sampled_close_points.shape[0]) * closest_points_to_q.shape[0])
        P = P[far_points_from_q]
        orig_indexes = orig_indexes[far_points_from_q]
        if not dist_matrix is None:
            dist_matrix = dist_matrix[far_points_from_q][:, far_points_from_q]
        # print(P.shape)
        # plt.plot(np.concatenate(coreset_list, axis=0)[:,0], np.concatenate(coreset_list, axis=0)[:,1], 'o', color = 'red',
        #     label="coreset")
        # plt.plot(P[:,0], P[:,1], '+', color = 'black',     label="data")
        # plt.show()
    # print(P.shape,coreset_iteration_sample_size)
    coreset_iteration_sample_size = min(P.shape[0], coreset_iteration_sample_size)
    last_idx = np.random.choice(P.shape[0], coreset_iteration_sample_size,
                                replace=Replace_in_coreset_sample)
    weights_list.append(np.ones(coreset_iteration_sample_size) * P.shape[0])
    coreset_list.append(P[last_idx])
    coreset_index_list.append(orig_indexes[last_idx])

    return coreset_list, weights_list, np.concatenate(coreset_list, axis=0), np.concatenate(weights_list, axis=0), \
        np.concatenate(coreset_index_list, axis=0)


def compute_robust_median(Q, tau, use_threshold_method, r, dist_matrix=None, is_max=False):
    opt_devided_by_size = None
    # print(Q.shape)
    distance_table = dist_matrix ** r if not dist_matrix is None else euclidean_distances(Q, Q) ** r
    distance_table.sort(axis=1)
    if is_max:
        sum_of_ditance_for_each_row = np.max(distance_table[:, :int(tau * Q.shape[0]) + 1], axis=1)
    else:
        sum_of_ditance_for_each_row = np.sum(distance_table[:, :int(tau * Q.shape[0]) + 1], axis=1)
    idx = np.argmin(sum_of_ditance_for_each_row)
    q = Q[idx]
    if use_threshold_method:
        opt_devided_by_size = sum_of_ditance_for_each_row[idx] / (int(tau * Q.shape[0]) + 1)
    return q, opt_devided_by_size, idx


def exp_runner(P, k=3, nqueries=1000, name_ext='',
               centers=None, mean_to_use=5,
               std_to_use=5,
               use_threshold_method=False,
               random_generation=False):
    n, d = P.shape
    coreset_results_list_avg = []
    unifrom_results_list_avg = []
    coreset_results_list_max = []
    unifrom_results_list_max = []
    coreset_results_list_opt = []
    unifrom_results_list_opt = []
    sizes_list = []
    if centers is None:
        centers = np.random.normal(mean_to_use, std_to_use, size=(nqueries, k, d))
    for coreset_iteration_sample_size in range(5, 50, 5):
        coreset_list, weights_list, coreset, weights = k_means_coreset_via_robust_median(P,
                                                                                         coreset_iteration_sample_size=coreset_iteration_sample_size,
                                                                                         k=k,
                                                                                         median_sample_size=500,
                                                                                         tau_for_the_sampled_set=None,
                                                                                         tau_for_the_original_set=None,
                                                                                         Replace_in_coreset_sample=False,
                                                                                         use_threshold_method=use_threshold_method,
                                                                                         random_generation=random_generation)

        weights = (weights * n) / np.sum(weights)

        unif_idx = np.random.choice(P.shape[0], coreset.shape[0])
        unif_sample = P[unif_idx]
        sizes_list.append(coreset.shape[0])
        ########################################################
        ########################################################
        k_means_for_coreset = compute_kmeans(coreset, k, weights=weights)
        k_means_for_unif = compute_kmeans(unif_sample, k)
        k_means_for_data = compute_kmeans(P, k)

        coreset_opt = k_means_cost(P, k_means_for_coreset)
        uniform_opt = k_means_cost(P, k_means_for_unif)
        data_opt = k_means_cost(P, k_means_for_data)

        coreset_opt_approx_error = np.abs(coreset_opt / data_opt - 1)
        unifrom_opt_approx_error = np.abs(uniform_opt / data_opt - 1)
        # print(coreset.shape)
        # plt.plot(coreset[:,0], coreset[:,1], 'o', color = 'red',
        #         label="coreset")
        # plt.plot(unif_sample[:,0], unif_sample[:,1], '^',color = 'blue',
        #        label="unifrom")
        # plt.show()
        # plt.plot(P[:,0], P[:,1], '+', color = 'black',
        #         label="P")
        # plt.show()
        #########################################################
        #########################################################
        c_coreset = 0;
        c_unif = 0;
        coreset_max_approx = 0;
        unif_max_approx = 0
        for i in range(nqueries):
            centers_to_check = centers[i]
            size_of_coreset = coreset.shape[0]
            data_cost = k_means_cost(P, centers_to_check)
            coreset_approx_error = np.abs(k_means_cost(coreset, centers_to_check, weights=weights) / data_cost - 1)
            unifrom_approx_error = np.abs(
                ((n / size_of_coreset) * (k_means_cost(unif_sample, centers_to_check))) / data_cost - 1)
            c_coreset += coreset_approx_error
            c_unif += unifrom_approx_error
            if coreset_max_approx < coreset_approx_error: coreset_max_approx = coreset_approx_error
            if unif_max_approx < unifrom_approx_error: unif_max_approx = unifrom_approx_error
        c_coreset = c_coreset / nqueries
        c_unif = c_unif / nqueries
        #########################################################
        #########################################################
        print("-------------------(n={},d={})------------------------".format(coreset.shape[0], coreset.shape[1]))
        print('Avarege Error: Coreset:={}, Uniform:={}'.format(c_coreset, c_unif))
        print('Maximum Error: Coreset:={}, Uniform:={}'.format(coreset_max_approx, unif_max_approx))
        print('Optimum Error: Coreset:={}, Uniform:={}'.format(coreset_opt_approx_error, unifrom_opt_approx_error))
        print("---------------------------------------------")
        coreset_results_list_avg.append(c_coreset)
        unifrom_results_list_avg.append(c_unif)
        coreset_results_list_max.append(coreset_max_approx)
        unifrom_results_list_max.append(unif_max_approx)
        coreset_results_list_opt.append(coreset_opt_approx_error)
        unifrom_results_list_opt.append(unifrom_opt_approx_error)
        #########################################################
        #########################################################
        # print(coreset.shape,len(coreset_list)* coreset_list[0].shape[0])
    np.save("coreset_results_avg_{}".format(name_ext), coreset_results_list_avg)
    np.save("uniform_results_avg_{}".format(name_ext), unifrom_results_list_avg)
    np.save("coreset_results_max_{}".format(name_ext), coreset_results_list_max)
    np.save("uniform_results_max_{}".format(name_ext), unifrom_results_list_max)
    np.save("coreset_results_opt_{}".format(name_ext), coreset_results_list_opt)
    np.save("unifrom_results_opt_{}".format(name_ext), unifrom_results_list_opt)
    np.save("coreset_sizes", sizes_list)


def run_multiple_exp(P=None, k=5, nqueries=1000, centers=None, usr_ext='normal',
                     repetitions=20, mean_to_use=5, std_to_use=5,
                     use_threshold_method=False, random_generation=False):
    if P is None:
        n, d = 20000, 4
        P = np.random.normal(5, 10, size=(n, d))

    for i in range(repetitions):
        ext = "{}_{}".format(usr_ext, i)
        # centers=P[np.random.choice(P.shape[0],nqueries,replace = False)]
        exp_runner(P, k=k, nqueries=nqueries, name_ext=ext,
                   centers=centers, mean_to_use=5, std_to_use=5,
                   use_threshold_method=use_threshold_method,
                   random_generation=random_generation)


def exp_plot(ext='normal', repetitions=20, dir_to_save='figs'):
    if not os.path.exists(dir_to_save): os.mkdir(dir_to_save)
    # sizes = [295, 520,720,900,1075,1230,1400,1560,1710] #list(range(295, 1710, 175))
    sizes = np.load("coreset_sizes.npy")
    sizes.sort()
    # sizes=sizes[::-1]
    print(sizes)
    exp_names = [['coreset_results_avg', 'uniform_results_avg'],
                 ['coreset_results_max', 'uniform_results_max'],
                 ['coreset_results_opt', 'unifrom_results_opt']]
    results_dict = {}

    for exp in flatten_list(exp_names):
        results_dict[exp] = {}
        results_dict[exp]['results_array'] = []
    for exp in flatten_list(exp_names):
        for i in range(repetitions):
            name_ext = "{}_{}.npy".format(ext, i)
            results_dict[exp]["results_array"].append(np.load("{}_{}".format(exp, name_ext)))
    for exp in flatten_list(exp_names):
        df_describe = pd.DataFrame(np.array(results_dict[exp]['results_array']))
        results_dict[exp]['desctibe'] = np.array(df_describe.describe())

    for pair_to_compare in exp_names:
        coreset = pair_to_compare[0]
        unifrom = pair_to_compare[1]
        for singe_statistic in ['mean', 'median']:
            coreset_array_to_plot = results_dict[coreset]['desctibe'][statistics_map(singe_statistic)]
            unifrom_array_to_plot = results_dict[unifrom]['desctibe'][statistics_map(singe_statistic)]

            stduniform = results_dict[unifrom]['desctibe'][statistics_map('std')]
            stdcoreset = results_dict[coreset]['desctibe'][statistics_map('std')]

            if singe_statistic == 'mean':
                plt.errorbar(sizes, coreset_array_to_plot, stdcoreset,
                             label='Coreset', color='red', fmt='-o')
                plt.errorbar(sizes, unifrom_array_to_plot, stduniform,
                             label='Uniform', color='blue', fmt='-^')
            else:
                plt.plot(sizes, coreset_array_to_plot, 'r-o',
                         label='Coreset', color='red')

                plt.plot(sizes, unifrom_array_to_plot, 'r-^',
                         label='Uniform', color='blue')

            plt.title('{} approximation error'.format(coreset.split("_")[-1]))
            plt.xlabel("Coreset size")
            plt.ylabel("Approximation error")
            # plt.xlim(0, 1)
            # plt.ylim(-5, 20)
            plt.legend(loc='best')
            plt.savefig('{}/{}_{}'.format(dir_to_save, coreset, singe_statistic))
            plt.show()
            plt.close()


"""features, true_labels = make_blobs(
    n_samples=10000,
    centers=7,
    cluster_std=5.75,
    random_state=42
)"""


# run_multiple_exp(features)
# exp_plot()
#############MY old checks##############

def distance_calculator_time_checker():
    for n in [50, 100, 200, 400, 800, 1600]:
        P = np.random.rand(n, 10)
        calculate_distances_using_different_methods(P)
    for d in [10, 20, 40, 80, 160, 320]:
        P = np.random.rand(800, d)
        calculate_distances_using_different_methods(P)


def calculate_distances_using_different_methods(Q):
    s = time.time()
    distarray = compute_dist_matrix_via_einsum(Q)
    e = time.time()
    print("-------------------(n={},d={})----------------------------".format(P.shape[0], P.shape[1]))
    print(e - s)
    s2 = time.time()
    distarray2 = euclidean_distances(Q, Q)
    e2 = time.time()
    print(e2 - s2)

    print(np.linalg.norm(distarray - distarray2))
    print('------------------------------------------------------------')


def compute_dist_matrix_via_einsum(a):
    b = a.reshape(a.shape[0], 1, a.shape[1])
    distarray = np.sqrt(np.einsum('ijk, ijk->ij', a - b, a - b))
    return distarray
