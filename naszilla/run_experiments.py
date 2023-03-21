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
from naszilla.params import *
from naszilla.nas_benchmarks import Nasbench101, Nasbench201, Nasbench301, KNasbench201
from naszilla.nas_algorithms import run_nas_algorithm

label_mapping = {'bananas': 'BANANAS', 'local_search': 'Local search', 'evolution': 'Evolutionary search',
                 'random': 'Random search'}


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

    def run_and_save(compression_method):
        cfg['compression_method'] = compression_method
        # manager = multiprocessing.Manager()
        algorithm_results = {}
        algorithm_val_results = {}
        k_lists = []
        q_lists = []
        results = {}
        val_results = {}
        walltimes = {}
        run_data = {}
        jobs = []

        if ss == 'nasbench_101':
            if args.k_alg:
                print('K alg not supported yet!')
                raise NotImplementedError()
            original_search_space = Nasbench101(mf=mf)
        elif ss == 'nasbench_201':
            original_search_space = Nasbench201(dataset=dataset) if not args.k_alg else \
                KNasbench201(dataset=dataset, dist_type=cfg['distance'], n_threads=cfg['threads'],
                             compression_method=compression_method,
                             compression_args=cfg['k_means_coreset_args'],
                             points_alg='evd')
        elif ss == 'nasbench_301':
            if args.k_alg:
                print('K alg not supported yet!')
                raise NotImplementedError()
            original_search_space = Nasbench301()
        else:
            print('Invalid search space')
            raise NotImplementedError()

        def trial(i, j):
            search_space = copy.deepcopy(original_search_space)

            starttime = time.time()
            # this line runs the nas algorithm and returns the result
            result, val_result, run_datum, cluster_sizes_list, kq_list = \
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
            walltimes[algorithm_params[j]['algo_name']][i] = time.time() - starttime
            results[algorithm_params[j]['algo_name']][i] = result
            val_results[algorithm_params[j]['algo_name']][i] = val_result
            run_data[algorithm_params[j]['algo_name']][i] = run_datum
            k_lists.append(np.array(kq_list[0]))
            q_lists.append(np.array(kq_list[1]))

        for j in range(num_algos):

            walltimes[algorithm_params[j]['algo_name']] = {}
            results[algorithm_params[j]['algo_name']] = {}
            val_results[algorithm_params[j]['algo_name']] = {}
            run_data[algorithm_params[j]['algo_name']] = {}
            print('\n* Running NAS algorithm: {}'.format(algorithm_params[j]))

            for i in range(trials//3):
                # p = multiprocessing.Process(target=trial, args=(i,j,))
                p1 = threading.Thread(target=trial, args=(i, j,))
                jobs.append(p1)
                p1.start()

                p2 = threading.Thread(target=trial, args=(i+1, j,))
                jobs.append(p2)
                p2.start()

                p3 = threading.Thread(target=trial, args=(i+2, j,))
                jobs.append(p3)
                p3.start()
                p1.join()
                p2.join()
                p3.join()

        # for proc in jobs:
        #     proc.join()
        #     time.sleep(2)

        for j in range(num_algos):

            tmp_results = list(results[algorithm_params[j]['algo_name']].values())
            tmp_val_results = list(val_results[algorithm_params[j]['algo_name']].values())
            tmp_walltimes = list(walltimes[algorithm_params[j]['algo_name']].values())
            tmp_run_data = list(run_data[algorithm_params[j]['algo_name']].values())

            if not len(tmp_results):
                continue

            # for idx in range(len(tmp_results)):
            #     for l in [tmp_results, tmp_val_results]:
            #         print(l)
            #         if isinstance(l[idx], np.ndarray):
            #             if not l[idx].size:
            #                 del l[idx]

            for i in range(len(tmp_results)):
                tmp_results[i] = tmp_results[i].T[1]
                tmp_val_results[i] = tmp_val_results[i].T[1]
            tmp_results = np.stack(tmp_results, axis=0)
            tmp_val_results = np.stack(tmp_val_results, axis=0)

            algorithm_results[algorithm_params[j]['algo_name']] = (
            np.mean(tmp_results, axis=0), np.std(tmp_results, axis=0))
            algorithm_val_results[algorithm_params[j]['algo_name']] = (
                np.mean(tmp_val_results, axis=0), np.std(tmp_val_results, axis=0))

            # print and pickle results
            filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
            print('\n* Trial summary: (params, results, walltimes)')
            print(algorithm_params)
            print(ss)
            print(results)
            print(tmp_walltimes)
            print('\n* Saving to file {}'.format(filename))
            with open(filename, 'wb') as f:
                pickle.dump([algorithm_params, metann_params, results, tmp_walltimes, tmp_run_data, val_results], f)
                f.close()

        if not os.path.exists('plots'):
            os.makedirs('plots')
        if not os.path.exists('plots/src_data'):
            os.makedirs('plots/src_data')
        for algo_name in algorithm_results.keys():
            result = 100 - algorithm_results[algo_name][0]
            val_result = 100 - algorithm_val_results[algo_name][0]
            np.save(
                'plots/src_data/{}_{}_{}_{}_val_mean'.format(cfg['figName'], args.dataset, compression_method,
                                                             algo_name),
                val_result)
            np.save(
                'plots/src_data/{}_{}_{}_{}_mean'.format(cfg['figName'], args.dataset, compression_method, algo_name),
                result)
            np.save(
                'plots/src_data/{}_{}_{}_{}_val_std'.format(cfg['figName'], args.dataset, compression_method,
                                                            algo_name),
                algorithm_val_results[algo_name][1])
            np.save(
                'plots/src_data/{}_{}_{}_{}_std'.format(cfg['figName'], args.dataset, compression_method, algo_name),
                algorithm_results[algo_name][1])

        return np.average(k_lists, axis=0).astype(int), np.average(q_lists, axis=0).astype(int)

    k_list, q_list = run_and_save('k_centers_coreset')
    # k_list, q_list = run_and_save('uniform')
    if args.study:
        cfg['kScheduler']['type'] = 'manual'
        cfg['kScheduler']['manual'] = k_list
        for compression_method in ['k_medians_coreset', 'k_centers_greedy',
                                   'k_means_coreset', 'k_medoids', 'uniform']:
            run_and_save(compression_method)


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
    parser.add_argument('--sample_size_graphs', type=int, default=0,
                        help='plot graphs with coreset size independent variable')
    parser.add_argument('--k_graphs', type=int, default=0, help='plot graphs with coreset size independent variable')
    parser.add_argument('--study', type=int, default=0, help='ablation study graphs')
    parser.add_argument('--cfg', type=str, default='/home/daniel/naszilla/naszilla/knas_config.yaml',
                        help='path to configuration file')

    args = parser.parse_args()
    main(args)
