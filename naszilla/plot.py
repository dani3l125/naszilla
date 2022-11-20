import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy
import yaml
import threading
import multiprocessing
from cycler import cycler
import matplotlib.pyplot as plt

from naszilla.params import *
from naszilla.nas_benchmarks import Nasbench101, Nasbench201, Nasbench301, KNasbench201
from naszilla.nas_algorithms import run_nas_algorithm

label_mapping = {'bananas':'BANANAS', 'local_search':'Local search', 'evolution':'Evolutionary search', 'random': 'Random search'}
name = 'm15_ciss1_k8_path_cifar100_k_means_coreset_orig_dist'
dataset = 'cifar100'
src_path = 'plots/src_data'

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
    custom_cycler = cycler(color=['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b', 'y', 'y', 'y'])
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
        if not os.path.exists(os.path.join(src_path, name+'_'+algo_name+'.npy')):
            break
        result = np.load(os.path.join(src_path, name+'_'+algo_name+'.npy'))
        val_result = np.load(os.path.join(src_path, name+'_'+algo_name+'_val.npy'))


        sota_result = 100 - np.load(f'/home/daniel/naszilla/sota_results/{algo_name}_{dataset}.npy')
        sota_val_result = 100 - np.load(f'/home/daniel/naszilla/sota_results/{algo_name}_{dataset}_val.npy')
        ax1.plot(np.arange(10, 301, 10), sota_result, 'o-', label=label_mapping[algo_name] + ', SOTA')
        ax1.plot(np.arange(10, 301, 10), result[9::10], '^-', label=label_mapping[algo_name] + ', ours')
        ax1.plot(x=np.arange(1, 301, 1), y=result,
                     fmt='-')
        ax2.plot(np.arange(10, 301, 10), sota_val_result, 'o-', label=label_mapping[algo_name] + ', SOTA')
        ax2.plot(np.arange(10, 301, 10), val_result[9::10], '^-', label=label_mapping[algo_name] + ', ours')
        ax2.plot(x=np.arange(1, 301, 1), y=val_result,
                     fmt='-')
    ax1.legend()
    ax2.legend()
    plt.savefig('plots/{}.png'.format(name))