import argparse
import sys
import os
import pickle
import numpy as np
import copy
import yaml
from cycler import cycler
import matplotlib.pyplot as plt

# label_mapping = {'bananas': 'BANANAS', 'local_search': 'Local search', 'evolution': 'Evolutionary search',
#                  'random': 'Random search'}
label_mapping = {'local_search': 'Local search', 'evolution': 'Evolutionary search',
                 'random': 'Random search'}

all_algs_mapping = {'k_centers_coreset': 'Coreset for k centers',
                    'k_centers_coreset_geometric': 'Coreset for k centers with geometric mapping',
                    'k_medians_coreset': 'Coreset for k medians', 'k_means_coreset': 'Coreset for k means',
                    'k_medoids': 'k medoids',
                    'uniform': 'Uniform sampling'}


def inverse(x):
    return x


def forward(x):
    return x


def plot_experiments(args):
    # read configuration
    cfg = yaml.safe_load(open(args.cfg, 'r')) if args.k_alg else None
    if args.study:
        compress_algs = all_algs_mapping.keys()
        color_list = ['r', 'g', 'b', 'y', 'c', 'm']
    else:
        compress_algs = ['k_centers_coreset']
        color_list = ['r', 'g']

    for algo_name in label_mapping.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        plt.title(label_mapping[algo_name])

        for compression_method in compress_algs:

            custom_cycler = cycler(color=color_list)
            cycle = custom_cycler()
            ax1.set_xlabel('Queries')
            ax1.set_ylabel('Best train accuracy')
            ax2.set_xlabel('Queries')
            ax2.set_ylabel('Best validation accuracy')

            ax1.set_yscale('function', functions=(forward, inverse))
            ax2.set_yscale('function', functions=(forward, inverse))
            if not os.path.exists('plots/src_data'):
                raise Exception("No source data")
            if not args.study:
                color = next(cycle)['color']
                # sota_result_mean = 100 - np.load(f'sota_results/{algo_name}_{args.dataset}_mean.npy')
                sota_val_result_mean = 100 - np.load(f'sota_results/{algo_name}_{args.dataset}_mean_val.npy')
                # sota_result_std = np.load(f'sota_results/{algo_name}_{args.dataset}_std.npy')
                sota_val_result_std = np.load(f'sota_results/{algo_name}_{args.dataset}_std_val.npy')
                # ax1.errorbar(np.arange(args.first, args.last + 1, 25), sota_result_mean[args.first - 1:args.last:25],
                #              yerr=sota_result_std[args.first - 1:args.last:25], fmt='.',
                #              label=f'{label_mapping[algo_name]}', color=color)
                # ax1.plot(np.arange(args.first, args.last + 1), sota_result_mean[args.first - 1:args.last], '-',
                #          color=color)

                ax2.errorbar(np.arange(args.first, args.last + 1, 25),
                             sota_val_result_mean[args.first - 1:args.last:25],
                             yerr=sota_val_result_std[args.first - 1:args.last:25], fmt='.',
                             label=f'{label_mapping[algo_name]}', color=color)
                ax2.plot(np.arange(args.first, args.last + 1), sota_val_result_mean[args.first - 1:args.last], '-',
                         color=color)

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

            color = next(cycle)['color']
            ax1.errorbar(np.arange(args.first, args.last + 1, 25), result_mean[args.first - 1:args.last:25],
                         yerr=result_std[args.first - 1:args.last:25], fmt='.',
                         label='{} + {}'.format(label_mapping[algo_name],
                                                'NASBoost' if not args.study else all_algs_mapping[compression_method]),
                         color=color)
            ax1.plot(np.arange(args.first, args.last + 1), result_mean[args.first - 1:args.last], '-',
                     color=color)

            ax2.errorbar(np.arange(args.first, args.last + 1, 25), val_result_mean[args.first - 1:args.last:25],
                         yerr=val_result_std[args.first - 1:args.last:25], fmt='.',
                         label='{} + {}'.format(label_mapping[algo_name],
                                                'NASBoost' if not args.study else all_algs_mapping[compression_method]),
                         color=color)
            ax2.plot(np.arange(args.first, args.last + 1), val_result_mean[args.first - 1:args.last], '-',
                     color=color)

        ax1.legend()
        ax2.legend()
        ax1.grid()
        ax2.grid()
        plt.savefig(
            'plots/{}_{}_{}_{}_{}.png'.format(cfg['figName'], args.dataset, algo_name, 'ablation' if args.study else '',
                                              f'query{args.first}to{args.last}'))


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
