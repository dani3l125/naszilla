import numpy as np
from naszilla.coresets.k_means_coreset_via_robust_median import k_means_coreset_via_robust_median
from naszilla.nas_benchmarks import KNasbench201
import os
from nas_201_api import NASBench201API as API
from naszilla.nas_bench_201.distances import *
from naszilla.nas_bench_201.cell_201 import Cell201
import argparse
import torch
import copy

is_debug = False

k_means_coreset_args = {'coreset_iteration_sample_size': 1,
                        'k': 50,
                        'k_ratio': 0,
                        'median_sample_size': 40,
                        'tau_for_the_sampled_set': None,
                        'tau_for_the_original_set': None,
                        'Replace_in_coreset_sample': 0,
                        'use_threshold_method': 0,
                        'random_generation': 0,
                        'r': 2}

parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
parser.add_argument('--dist', type=str, default='lev', help='Number of trials')
args = parser.parse_args()

k_list = [4, 6, 8, 10, 12, 14]
coreset_iteration_sample_size_list = [1]
median_sample_size_list = [20]

DEFAULT_K = 12
DEFAULT_CISS = 1
DEFAULT_SAMPLE_SIZE_LIST = 20
data_folder = '~/nas_benchmark_datasets/'

if is_debug:
    search_space = torch.load(os.path.expanduser(data_folder + 'NAS-Bench-mini.pth'))
elif version == '1_0':
    search_space = API(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_0-e61699.pth'))
archs_val = np.zeros((3, len(search_space)))
for i in range(len(search_space)):
    archs_val[0, i] = search_space.query_by_index(i).get_metrics('cifar10-valid', 'x-valid')['accuracy']
    archs_val[1, i] = search_space.query_by_index(i).get_metrics('cifar100', 'x-valid')['accuracy']
    archs_val[2, i] = search_space.query_by_index(i).get_metrics('ImageNet16-120', 'x-valid')['accuracy']

d = 10
dataset = 'cifar100'
dist_matrix = np.load(f'distances/{args.dist}_dist.npy')
P = np.zeros_like(dist_matrix)[:, :d]

distace_functions = {
    'adj': adj_distance,
    'path': path_distance,
    'lev': lev_distance,
    'nasbot': nasbot_distance,
    #    'real': real_distance
}

def calculate_distance_mat(dist_name):
    if os.path.isfile(f'distances/{dist_name}_dist.npy'):
        return np.load(f'distances/{dist_name}_dist.npy')
    dist_matrix = np.zeros((len(search_space), len(search_space)))
    for i, str1 in enumerate(search_space.meta_archs):
        for j, str2 in enumerate(search_space.meta_archs):
            if dist_name == 'real':
                dist_matrix[i, j] = real_distance(Cell201(str1), Cell201(str2), search_space)
            else:
                dist_matrix[i, j] = distace_functions[dist_name](Cell201(str1), Cell201(str2))

    np.save(f'distances/{dist_name}_dist.npy', dist_matrix)
    return dist_matrix

def coreset_stats(k, coreset_iteration_sample_size, median_sample_size, num_of_optimums=30):
    _, _, coreset, _, coreset_indexes = k_means_coreset_via_robust_median(P, dist_matrix, k=k,
                                                                          coreset_iteration_sample_size=coreset_iteration_sample_size,
                                                                          median_sample_size=median_sample_size)

    points2coreset_dist_mat = dist_matrix[:, coreset_indexes]
    labels = np.argmin(points2coreset_dist_mat, axis=1)
    distances_to_representatives = np.min(points2coreset_dist_mat, axis=1)
    sizes = np.bincount(labels)
    mean = np.mean(sizes)
    std = np.std(sizes)

    print(
        f'\nk = {k} | coreset iteration sample size = {coreset_iteration_sample_size} | median sample size = {median_sample_size}')
    print(f'coreset size = {coreset_indexes.size}')
    print(f'cluster size:(mean, std) = ({mean}. {std})')

    for i, dataset in enumerate(['cifar10', 'cifar100', 'ImageNet16-120']):
        print(f'\n\tDataset:{dataset}')
        optimum_indexes = np.flip(np.argsort(archs_val[i]))[:num_of_optimums]
        print(f'\t{num_of_optimums} best architectures validation accuracy values: {archs_val[i][optimum_indexes]}')
        optimum_labels = labels[optimum_indexes]
        representatives_indexes = coreset_indexes[optimum_labels]
        optimum_accuracy_dists = np.zeros_like(optimum_labels)
        # for j in range(optimum_labels.size):
        #     optimum_accuracy_dists[j] = np.abs(archs_val[i, optimum_indexes[j]] - archs_val[i, representatives_indexes[j]])
        optimum_accuracy_dists = np.abs(archs_val[i, optimum_indexes] - archs_val[i, representatives_indexes])
        optimum_euclidean_dists = dist_matrix[optimum_indexes, representatives_indexes]
        print(
            f'\t{num_of_optimums} best architectures validation accuracy distance from representative: {optimum_accuracy_dists}')
        print(
            f'\t{num_of_optimums} best architectures {args.dist} (pseudo euclidean) distance from representative: {optimum_euclidean_dists}')
        cluster_accuracy_distance = np.zeros(optimum_labels.size)
        cluster_euclidean_distance = np.zeros(optimum_labels.size)
        for idx, label in enumerate(optimum_labels):
            label_indexes = np.where(labels == label)
            cluster_accuracy_distance[idx] = np.mean(
                np.abs(archs_val[i, label_indexes] - archs_val[i, representatives_indexes[idx]]))
            cluster_euclidean_distance[idx] = np.mean(dist_matrix[representatives_indexes[idx], label_indexes])
        print(f'\t{num_of_optimums} best architecture\'s clusters:')
        print(f'\tcluster sizes: {sizes[optimum_labels]}')
        print(f'\tAverage accuracy distance from representative (in cluster): {cluster_accuracy_distance}')
        print(
            f'\tAverage {args.dist} (euclidean) distance from representative (in cluster): {cluster_euclidean_distance}')

        #
        # optimum_indexes = np.flip(np.argsort(archs_val[i]))
        # best_values_indexes = np.zeros((optimum_indexes.size))
        # cluster_best_vals = np.zeros(coreset_indexes.size) * 10000
        # cluster_representative_vals = np.zeros(coreset_indexes.size)
        # for arch_idx, label in enumerate(labels):
        #     if archs_val[i, arch_idx] > cluster_best_vals[label]:
        #         best_values_indexes[label] = arch_idx
        #         cluster_best_vals[label] = max(cluster_best_vals[label],  archs_val[i, arch_idx])
        # dist = 0
        # count = 0
        # for label in range(coreset_indexes.size):
        #     cluster_representative_vals[label] = archs_val[i, coreset_indexes[label]]
        #     if cluster_best_vals[label] != 0:
        #         dist += np.abs(cluster_best_vals[label] - cluster_representative_vals[label])
        #         count += 1
        #     # if label%100 == 0:
        #     #     print(f'Dataset = {dataset} | Cluster = {label} | Cluster representative value = {cluster_representative_vals[label]} | Cluster best value = {cluster_best_vals[label]} | Difference = {np.abs(cluster_best_vals[label] - cluster_representative_vals[label])}')
        # dist /= count
        # print(f'Avarage Distance = {dist}')
        # print(f'Avarage distance from point to representative = {np.mean(distances_to_representatives)}')
        # #print(f'Maximum distance between best arch and representative = {np.max(dist_matrix[coreset_indexes.astype(int), best_values_indexes.astype(int)])}')
        # print(f'Accuracy distances between best architectures and its representatives:{optimum_accuracy_dists}')
        # print(f'Euclidean distances between best architectures and its representatives:{optimum_euclidean_dists}')


def search_space_stats(num_of_optimums=30, knn=150):
    for i, dataset in enumerate(['cifar10', 'cifar100', 'ImageNet16-120']):
        optimum_indexes = np.argsort(archs_val[i])[::-1][:num_of_optimums]
        knn_arr = np.argsort(dist_matrix[optimum_indexes])[:, :knn]
        for j, optimum in enumerate(optimum_indexes):
            knn_vals = []
            for neighbour in knn_arr[j]:
                knn_vals.append(np.abs(archs_val[i, optimum] - archs_val[i, neighbour]))
            print(f'{dataset} Dataset | {j} optimum | {knn_vals}')


def cluster_accuracy_statistics(space, dist_matrix):
    for i, dataset in enumerate(['cifar10', 'cifar100', 'ImageNet16-120']):
        clusters_num = space.labels.max() + 1
        clusters_best_values = -1 * np.ones((clusters_num,))
        clusters_maximal_distances_acc = -1 * np.ones((clusters_num,))
        clusters_center_distances_acc = -1 * np.ones((clusters_num,))
        clusters_maximal_distances_metric = -1 * np.ones((clusters_num,))
        clusters_center_distances_metric = -1 * np.ones((clusters_num,))
        evaluated_indexes_array = np.array(space.old_nasbench.evaluated_indexes)
        for label in range(clusters_num):
            label_values = archs_val[i][evaluated_indexes_array
            [np.argwhere(space.labels == label)].T[0]]
            clusters_best_values[label] = label_values.max()
            clusters_maximal_distances_acc[label] = clusters_best_values[label] - label_values.min()
            clusters_center_distances_acc[label] = clusters_best_values[label] - archs_val[i][
                space.coreset_indexes[label]]
            clusters_maximal_distances_metric[label] = dist_matrix[evaluated_indexes_array
            [np.argwhere(space.labels == label)].T[0]][:, evaluated_indexes_array
                                                     [np.argwhere(space.labels == label)].T[0]].max()
            clusters_center_distances_metric[label] = dist_matrix[evaluated_indexes_array
            [np.argwhere(space.labels == label)].T[0]][:, space.coreset_indexes[label]].max()

        return {'clusters maximal accuracy distance (mean, std)': (
                np.mean(clusters_maximal_distances_acc), np.std(clusters_maximal_distances_acc)),
                'clusters maximal accuracy distance from center (mean, std)': (
                np.mean(clusters_center_distances_acc), np.std(clusters_center_distances_acc)),
                'clusters maximal metric distance (mean, std)': (
                np.mean(clusters_maximal_distances_metric), np.std(clusters_maximal_distances_metric)),
                'clusters maximal metric distance from center (mean, std)': (
                np.mean(clusters_center_distances_metric), np.std(clusters_center_distances_metric))
                }




if __name__ == '__main__':
    statistics_dict = {}
    for dist_name in distace_functions.keys():
        print(f'Calculating {dist_name} distance...')
        dist_matrix = calculate_distance_mat(dist_name)
        print('Done!')
        statistics_dict[dist_name] = {}
        for compression_method in ['k_means_coreset_orig_dist', 'k_means_coreset', 'uniform', 'k_medoids']:
            space = copy.deepcopy(search_space)
            if is_debug:
                dist_matrix = dist_matrix[:150][:,:150]
            space.prune(0, 400)
            statistics_dict[dist_name][compression_method] = cluster_accuracy_statistics(space, dist_matrix)

    print(statistics_dict)
    torch.save(statistics_dict, 'statistics.pth')
    print('Statistics saved')
