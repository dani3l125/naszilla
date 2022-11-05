import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy
import yaml
import multiprocessing
from cycler import cycler
import matplotlib.pyplot as plt

from naszilla.params import *
from naszilla.nas_benchmarks import Nasbench101, Nasbench201, Nasbench301, KNasbench201
from naszilla.nas_algorithms import run_nas_algorithm


def run_experiments(args, save_dir):
    # set up arguments
    trials = args.trials
    queries = args.queries
    out_file = args.output_filename
    save_specs = args.save_specs
    metann_params = meta_neuralnet_params(args.metann_params)
    ss = args.search_space
    dataset = args.dataset
    mf = args.mf
    algorithm_params = algo_params(args.algo_params, queries=queries)
    num_algos = len(algorithm_params)
    logging.info(algorithm_params)

    # set up search space
    mp = copy.deepcopy(metann_params)

    # read configuration
    cfg = yaml.safe_load(open(args.cfg, 'r')) if args.k_alg else None

    for compression_method in ['k_means_coreset_orig_dist', 'k_means_coreset', 'uniform', 'k_medoids']:

        manager = multiprocessing.Manager()
        algorithm_results = manager.dict()
        algorithm_val_results = manager.dict()
        jobs = []

        for j in range(num_algos):
            print('\n* Running NAS algorithm: {}'.format(algorithm_params[j]))

            results = manager.dict()
            val_results = manager.dict()
            walltimes = manager.dict()
            run_data = manager.dict()

            def trial(i):
                if ss == 'nasbench_101':
                    if args.k_alg:
                        print('K alg not supported yet!')
                        raise NotImplementedError()
                    search_space = Nasbench101(mf=mf)
                elif ss == 'nasbench_201':
                    search_space = Nasbench201(dataset=dataset) if not args.k_alg else \
                        KNasbench201(dataset=dataset, dist_type=cfg['distance'], n_threads=cfg['threads'],
                                     compression_method=compression_method,
                                     compression_args=cfg['k_means_coreset_args'],
                                     points_alg='evd')
                elif ss == 'nasbench_301':
                    if args.k_alg:
                        print('K alg not supported yet!')
                        raise NotImplementedError()
                    search_space = Nasbench301()
                else:
                    print('Invalid search space')
                    raise NotImplementedError()

                starttime = time.time()
                # this line runs the nas algorithm and returns the result
                result, val_result, run_datum, cluster_sizes_list = \
                    run_nas_algorithm(algorithm_params[j], search_space, mp, args.k_alg, cfg)

                result = np.round(result, 5)
                val_result = np.round(val_result, 5)

                # remove unnecessary dict entries that take up space
                for d in run_datum:
                    if not save_specs:
                        d.pop('spec')
                    for key in ['encoding', 'adj', 'path', 'dist_to_min']:
                        if key in d:
                            d.pop(key)

                # add walltime, results, run_data
                walltimes[i] = time.time() - starttime
                results[i] = result
                val_results[i] = val_result
                run_data[i] = run_datum

            for i in range(trials):
                p = multiprocessing.Process(target=trial, args=(i,))
                jobs.append(p)
                p.start()

            tmp_results = list(results.values())
            tmp_val_results = list(val_results.values())
            walltimes = list(walltimes.values())
            run_data = list(run_data.values())

            for i in range(len(tmp_results)):
                tmp_results[i] = tmp_results[i].T[1]
                tmp_val_results[i] = tmp_val_results[i].T[1]
            tmp_results = np.stack(results, axis=0)
            tmp_val_results = np.stack(val_results, axis=0)

            algorithm_results[algorithm_params[j]['algo_name']] = (np.mean(tmp_results, axis=0), np.std(tmp_results, axis=0))
            algorithm_val_results[algorithm_params[j]['algo_name']] = (
            np.mean(tmp_val_results, axis=0), np.std(tmp_val_results, axis=0))

            # print and pickle results
            filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
            print('\n* Trial summary: (params, results, walltimes)')
            print(algorithm_params)
            print(ss)
            print(results)
            print(walltimes)
            print('\n* Saving to file {}'.format(filename))
            with open(filename, 'wb') as f:
                pickle.dump([algorithm_params, metann_params, results, walltimes, run_data, val_results], f)
                f.close()

        for proc in jobs:
            proc.join()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        custom_cycler = cycler(color=['r', 'r', 'g', 'g', 'b', 'b', 'y', 'y'])
        ax1.set_xlabel('Queries')
        ax1.set_ylabel('Best accuracy')
        ax2.set_xlabel('Queries')
        ax2.set_ylabel('Best accuracy')
        ax1.set_prop_cycle(custom_cycler)
        ax2.set_prop_cycle(custom_cycler)
        for algo_name in algorithm_results.keys():
            sota_result = 100 - np.load(f'sota_results/{algo_name}_{args.dataset}.npy')
            sota_val_result = 100 - np.load(f'sota_results/{algo_name}_{args.dataset}_val.npy')
            result = 100 - algorithm_results[algo_name][0]
            val_result = 100 - algorithm_val_results[algo_name][0]
            ax1.plot(np.arange(10, 301, 10), sota_result, '--')
            ax1.plot(np.arange(1, 301, 1), result, '^-')
            ax1.plot(np.arange(10, 301, 10), sota_val_result, '--')
            ax1.plot(np.arange(1, 301, 1), val_result, '^-')
            np.save(
                'plots/src_data/{}_{}_{}_{}_val.png'.format(cfg['figName'], args.dataset, compression_method, algo_name),
                val_result)
            np.save(
                'plots/src_data/{}_{}_{}_{}.png'.format(cfg['figName'], args.dataset, compression_method, algo_name),
                result)
        plt.savefig('plots/{}_{}_{}.png'.format(cfg['figName'], args.dataset, compression_method))


def main(args):
    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists('sota_results'):
        os.mkdir('sota_results')

    algo_params = args.algo_params
    save_path = save_dir + '/' + algo_params + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_path)


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
    parser.add_argument('--sample_size_graphs', type=int, default=0, help='plot graphs with coreset size independent variable')
    parser.add_argument('--k_graphs', type=int, default=0, help='plot graphs with coreset size independent variable')
    parser.add_argument('--cfg', type=str, default='/home/daniel/naszilla/naszilla/knas_config.yaml',
                        help='path to configuration file')

    args = parser.parse_args()
    main(args)
