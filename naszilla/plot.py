import argparse
import sys
import os
import pickle
import numpy as np
import copy
import yaml
from cycler import cycler
import matplotlib.pyplot as plt

label_mapping = {'bananas': 'BANANAS', 'local_search': 'Local search', 'evolution': 'Evolutionary search',
                 'random': 'Random search'}

all_algs = ['k_centers_coreset_geometric', 'k_medians_coreset', 'k_means_coreset', 'k_medoids', 'uniform']


def inverse(x):
    return x


def forward(x):
    return x


def plot_experiments(args):
    # read configuration
    cfg = yaml.safe_load(open(args.cfg, 'r')) if args.k_alg else None
    if args.study:
        compress_algs = all_algs
        color_list = ['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'y', 'y', 'y']
    else:
        compress_algs = ['k_means_coreset_orig_dist']
        color_list = ['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'y', 'y', 'y']

    for algo_name in label_mapping.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        plt.title(label_mapping[algo_name])

        for compression_method in compress_algs:

            custom_cycler = cycler(color=color_list)
            ax1.set_xlabel('Queries')
            ax1.set_ylabel('Best train accuracy')
            ax2.set_xlabel('Queries')
            ax2.set_ylabel('Best validation accuracy')
            ax1.set_prop_cycle(custom_cycler)
            ax2.set_prop_cycle(custom_cycler)

            ax1.set_yscale('function', functions=(forward, inverse))
            ax2.set_yscale('function', functions=(forward, inverse))
            if not os.path.exists('plots/src_data'):
                raise Exception("No source data")
            if not args.study:
                sota_result_mean = 100 - np.load(f'sota_results/{algo_name}_{args.dataset}_mean.npy')
                sota_val_result_mean = 100 - np.load(f'sota_results/{algo_name}_{args.dataset}_mean_val.npy')
                sota_result_std = np.load(f'sota_results/{algo_name}_{args.dataset}_std.npy')
                sota_val_result_std = np.load(f'sota_results/{algo_name}_{args.dataset}_std_val.npy')

            if not os.path.exists(
                    'plots/src_data/{}_{}_{}_{}_mean.npy'.format(cfg['figName'], args.dataset, compression_method,
                                                                 algo_name)):
                break
            result_mean = np.load(
                'plots/src_data/{}_{}_{}_{}_mean.npy'.format(cfg['figName'], args.dataset, compression_method,
                                                             algo_name))
            val_result_mean = np.load(
                'plots/src_data/{}_{}_{}_{}_val_mean.npy'.format(cfg['figName'], args.dataset, compression_method,
                                                                 algo_name))
            result_std = np.load(
                'plots/src_data/{}_{}_{}_{}_std.npy'.format(cfg['figName'], args.dataset, compression_method,
                                                            algo_name))
            val_result_std = np.load(
                'plots/src_data/{}_{}_{}_{}_val_std.npy'.format(cfg['figName'], args.dataset, compression_method,
                                                                algo_name))
            ax1.errorbar(np.arange(args.first, args.last, 50), result_mean[args.first - 1:args.last:50],
                         yerr=result_std[args.first - 1:args.last:50], fmt='*',
                         label=f'NASBoost + {label_mapping[algo_name]}')
            ax1.plot(np.arange(args.first, args.last), val_result_mean[args.first - 1:args.last], '-')

            ax2.errorbar(np.arange(args.first, args.last, 50), val_result_mean[args.first - 1:args.last:50],
                         yerr=val_result_std[args.first - 1:args.last:50], fmt='*',
                         label=f'NASBoost + {label_mapping[algo_name]}')
            ax2.plot(np.arange(args.first, args.last), result_mean[args.first - 1:args.last], '-')

        ax1.legend()
        ax2.legend()
        plt.savefig('plots/{}_{}_{}.png'.format(cfg['figName'], args.dataset, algo_name))


def main(args):
    plot_experiments(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
    parser.add_argument('--trials', type=int, default=500, help='Number of trials')
    parser.add_argument('--queries', type=int, default=150,
                        help='Max number of queries/evaluations each NAS algorithm gets')
    parser.add_argument('--search_space', type=str, default='nasbench_101', help='nasbench_101, _201, or _301')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, 100, or imagenet (for nasbench201)')
    parser.add_argument('--mf', type=bool, default=False, help='Multi fidelity: true or false (for nasbench101)')
    parser.add_argument('--metann_params', type=str, default='standard', help='which parameters to use')
    parser.add_argument('--algo_params', type=str, default='simple_algos', help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default='round', help='name of output files')
    parser.add_argument('--save_dir', type=str, default='results_output', help='name of save directory')
    parser.add_argument('--save_specs', type=bool, default=False, help='save the architecture specs')
    parser.add_argument('--save_sota', type=int, default=0, help='save the convergence result to a numpy array')
    parser.add_argument('--k_alg', type=int, default=0, help='use iterative k algorithm')
    parser.add_argument('--study', type=int, default=0, help='ablation study graphs')
    parser.add_argument('--sample_size_graphs', type=int, default=0,
                        help='plot graphs with coreset size independent variable')
    parser.add_argument('--k_graphs', type=int, default=0, help='plot graphs with coreset size independent variable')
    parser.add_argument('--first', type=int, default=0, help='first query forx x axis')
    parser.add_argument('--last', type=int, default=0, help='last query forx x axis')
    parser.add_argument('--cfg', type=str, default='/home/daniel/naszilla/naszilla/knas_config.yaml',
                        help='path to configuration file')

    args = parser.parse_args()
    main(args)
