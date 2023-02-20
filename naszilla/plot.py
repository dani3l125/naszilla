import argparse
import sys
import os
import pickle
import numpy as np
import copy
import yaml
from cycler import cycler
import matplotlib.pyplot as plt


label_mapping = {'bananas':'BANANAS', 'local_search':'Local search', 'evolution':'Evolutionary search', 'random': 'Random search'}

def plot_experiments(args):
    # read configuration
    cfg = yaml.safe_load(open(args.cfg, 'r')) if args.k_alg else None
    if args.study:
        compress_algs = ['k_means_coreset_orig_dist', 'k_means_coreset', 'uniform', 'k_medoids']
        color_list = ['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'y', 'y', 'y']
    else:
        compress_algs = ['k_means_coreset_orig_dist']
        color_list = ['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'y', 'y', 'y']
    for compression_method in compress_algs:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        custom_cycler = cycler(color=color_list)
        ax1.set_xlabel('Queries')
        ax1.set_ylabel('Best train accuracy')
        ax2.set_xlabel('Queries')
        ax2.set_ylabel('Best validation accuracy')
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        def inverse(x):
            return x
        def forward(x):
            return x
        ax1.set_yscale('function', functions=(forward, inverse))
        ax2.set_yscale('function', functions=(forward, inverse))
        if not os.path.exists('plots'):
            os.makedirs('plots')
        if not os.path.exists('plots/src_data'):
            os.makedirs('plots/src_data')
        for algo_name in label_mapping.keys():
            sota_result = 100 - np.load(f'sota_results_old/{algo_name}_{args.dataset}.npy')
            sota_val_result = 100 - np.load(f'sota_results_old/{algo_name}_{args.dataset}_val.npy')
            if not os.path.exists('plots/src_data/{}_{}_{}_{}.npy'.format(cfg['figName'], args.dataset, compression_method, algo_name)):
                break
            result = np.load(
                'plots/src_data/{}_{}_{}_{}.npy'.format(cfg['figName'], args.dataset, compression_method, algo_name))
            val_result = np.load(
                'plots/src_data/{}_{}_{}_{}_val.npy'.format(cfg['figName'], args.dataset, compression_method, algo_name))
            ax1.plot(np.arange(10, 301, 50), result[9::50], '^', label=label_mapping[algo_name] + ', ours')
            ax1.plot(np.arange(1, 301, 1)[9:], result[9:], '-')
            ax2.plot(np.arange(10, 301, 50), val_result[9::50], '^', label=label_mapping[algo_name]+', ours')
            ax2.plot(np.arange(1, 301, 1)[9:], val_result[9:], '-')

        ax1.legend()
        ax2.legend()
        plt.savefig('plots/{}_{}_{}.png'.format(cfg['figName'], args.dataset, compression_method))


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
    parser.add_argument('--sample_size_graphs', type=int, default=0, help='plot graphs with coreset size independent variable')
    parser.add_argument('--k_graphs', type=int, default=0, help='plot graphs with coreset size independent variable')
    parser.add_argument('--first', type=int, default=0, help='first query forx x axis')
    parser.add_argument('--last', type=int, default=0, help='last query forx x axis')
    parser.add_argument('--cfg', type=str, default='/home/daniel/naszilla/naszilla/knas_config.yaml',
                        help='path to configuration file')

    args = parser.parse_args()
    main(args)
