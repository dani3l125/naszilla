import itertools
import os
import pickle
import sys
import copy
import gc
import numpy as np
import tensorflow as tf
from argparse import Namespace

from naszilla.acquisition_functions import acq_fn
from naszilla.meta_neural_net import MetaNeuralnet
from naszilla.bo.bo.probo import ProBO
from naszilla.gcn.model import NeuralPredictor
from naszilla.gcn.train_gcn import fit, predict
from nas_benchmarks import MutationTree

# default parameters for the NAS algorithms
DEFAULT_NUM_INIT = 10
DEFAULT_K = 10
DEFAULT_TOTAL_QUERIES = 150
DEFAULT_LOSS = 'val_loss'


def run_nas_algorithm(algo_params, search_space, mp, k_alg, cfg):
    # run nas algorithm
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    cluster_sizes_list = []

    if k_alg:
        if cfg is None:
            raise Exception('No configuration file')
        data = knas(algo_params, search_space, mp, cfg)
    elif algo_name == 'random':
        data = random_search(search_space, **ps)
    elif algo_name == 'evolution':
        data = evolution_search(search_space, **ps)
    elif algo_name == 'bananas':
        data = bananas(search_space, mp, **ps, predictor='bananas')
    elif algo_name == 'bonas':
        data = bananas(search_space, mp, **ps, predictor='gcn', predictor_encoding='gcn')
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_search(search_space, **ps)
    elif algo_name == 'dngo':
        data = pybnn_search(search_space, model_type='dngo', **ps)
    elif algo_name == 'bohamiann':
        data = pybnn_search(search_space, model_type='bohamiann', **ps)
    elif algo_name == 'local_search':
        data = local_search(search_space, **ps)
    elif algo_name == 'gcn_predictor':
        data = gcn_predictor(search_space, **ps)
    elif algo_name == 'vaenas':
        # data = bananas(search_space, mp, **ps, predictor='vae', predictor_encoding='vae')
        print('Currently not implemented')
        raise NotImplementedError()
    else:
        print('Invalid algorithm name')
        raise NotImplementedError()

    if 'k' not in ps:
        ps['k'] = DEFAULT_K
    if 'total_queries' not in ps:
        ps['total_queries'] = DEFAULT_TOTAL_QUERIES
    if 'loss' not in ps:
        ps['loss'] = DEFAULT_LOSS

    result, val_result = compute_best_test_losses(data, DEFAULT_K, ps['total_queries'], DEFAULT_LOSS)
    return result, val_result, data, cluster_sizes_list


def schedule_linear(first, last, n):
    d = (last - first) / (n - 1)
    if d == 0:
        return first * (np.ones(n)).astype(int)
    return np.round(np.arange(first, last + d, d)).astype(int)


def schedule_geometric(first, last, n):
    return np.round(np.geomspace(first, last, n)).astype(int)


def schedule_linear_by_sum(first, s, n):
    d = 2 * (((s / n) - first) / (n - 1))
    above_last = first + d * n
    if d == 0:
        return first * (np.ones(n)).astype(int)
    return np.round(np.arange(first, above_last, d)).astype(int)


def schedule_geometric_by_sum(ratio, s, n):
    r_n = ratio ** n
    first = (s * (1 - ratio)) / (1 - r_n)
    return np.round(np.geomspace(first, first * r_n / ratio, n))[::-1].astype(int)


def knas(algo_params, search_space, mp, cfg):
    # run nas algorithm

    n_iterations = cfg['iterations']
    total_q = algo_params['total_queries']
    q_sum = 0
    space_size = len(search_space)

    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    final_data = []

    if algo_name == 'bananas':
        meta_neuralnet = MetaNeuralnet()

    for i in range(n_iterations):
        # Case of checking complete space:
        if (space_size <= total_q - q_sum and cfg['qAutofill']) or (cfg['compression_method'] == 'k_means_coreset' and space_size < 30):
            k = -1
            q = space_size
            m = -1
        else:
            if cfg['kScheduler']['type'] == 'log':
                k = int(np.ceil(np.log2(space_size)))
                q = k
                m = int(max(1, np.round(k/cfg['m_c'])))
            elif cfg['kScheduler']['type'] == 'geometric':
                k = len(search_space) // cfg['kScheduler']['constant']
                q = min(30, k)  # TODO: schedule q properly (for efficiency)
                m = q // cfg['m_c']
            elif cfg['kScheduler']['type'] == 'manual':
                k = min(cfg['kScheduler']['manual'][i], len(search_space))
                q = min(30, k)  # TODO: schedule q properly (for efficiency)
                m = q // cfg['m_c']

        if k != -1:
            # -1 means searching in the final cluster as usual
            k = search_space.prune(k)
            if cfg['compression_method'] == 'k_means_coreset':
                q = min(30, k)  # TODO: schedule q properly (for efficiency)
                m = q // cfg['m_c']


        ps['total_queries'] = q
        q_sum += q
        print(f'#####\nIteration {i + 1}: k = {k}; m = {m}; q = {q}')

        if algo_name == 'pknas':
            data = search_space.generate_complete_dataset()
        elif algo_name == 'random':
            data = random_search(search_space, **ps)
        elif algo_name == 'evolution':
            data = evolution_search(search_space, **ps)
        elif algo_name == 'bananas':
            data = bananas(search_space, mp, **ps, predictor='bananas',
                           mutation_tree=MutationTree(search_space.nasbench.meta_archs),
                           meta_neuralnet=meta_neuralnet)
        elif algo_name == 'bonas':
            data = bananas(search_space, mp, **ps, predictor='gcn', predictor_encoding='gcn')
        elif algo_name == 'gp_bayesopt':
            data = gp_bayesopt_search(search_space, **ps)
        elif algo_name == 'dngo':
            data = pybnn_search(search_space, model_type='dngo', **ps)
        elif algo_name == 'bohamiann':
            data = pybnn_search(search_space, model_type='bohamiann', **ps)
        elif algo_name == 'local_search':
            data = local_search(search_space, **ps)
        elif algo_name == 'gcn_predictor':
            data = gcn_predictor(search_space, **ps)
        elif algo_name == 'vaenas':
            # data = bananas(search_space, mp, **ps, predictor='vae', predictor_encoding='vae')
            print('Currently not implemented')
            raise NotImplementedError()
        else:
            print('Invalid algorithm name')
            raise NotImplementedError()
        final_data.extend(data)
        # if algo_name == 'pknas':
        _, val_result = compute_best_test_losses(data, DEFAULT_K, ps['total_queries'], DEFAULT_LOSS)
        print(f'\n Result: {val_result} Optimal: {search_space.get_best_arch_loss()}\n#####')

        if k != -1:  # efficiency
            search_space.choose_clusters(data, int(m))
        else:
            return final_data
        space_size = len(search_space)

    return final_data


def compute_best_test_losses(data, k, total_queries, loss):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error 
    after every multiple of k
    """
    results = []
    val_results = []
    if len(data) == 0:
        print('Result output is empty.')
        return
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i: i[loss])[0]
        val_error = best_arch['val_loss']
        test_error = best_arch['test_loss']
        results.append((query, test_error))
        val_results.append((query, val_error))

    return results, val_results



def random_search(search_space,
                  total_queries=DEFAULT_TOTAL_QUERIES,
                  loss=DEFAULT_LOSS,
                  random_encoding='adj',
                  cutoff=0,
                  deterministic=True,
                  verbose=1):
    """ 
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries,
                                                random_encoding=random_encoding,
                                                cutoff=cutoff,
                                                deterministic_loss=deterministic)

    if verbose:
        top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
        print('random, query {}, top 5 losses {}'.format(total_queries, top_5_loss))
    return data


def evolution_search(search_space,
                     total_queries=DEFAULT_TOTAL_QUERIES,
                     num_init=DEFAULT_NUM_INIT,
                     k=DEFAULT_K,
                     loss=DEFAULT_LOSS,
                     population_size=30,
                     tournament_size=10,
                     mutation_rate=1.0,
                     mutate_encoding='adj',
                     cutoff=0,
                     random_encoding='adj',
                     deterministic=True,
                     regularize=True,
                     verbose=1):
    """
    regularized evolution
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                random_encoding=random_encoding,
                                                deterministic_loss=deterministic)

    losses = [d[loss] for d in data]
    query = num_init
    population = [i for i in range(min(num_init, population_size))]

    while query <= total_queries:

        # evolve the population by mutating the best architecture
        # from a random subset of the population
        sample = np.random.choice(population, tournament_size)
        best_index = sorted([(i, losses[i]) for i in sample], key=lambda i: i[1])[0][0]
        mutated = search_space.mutate_arch(data[best_index]['spec'],
                                           mutation_rate=mutation_rate,
                                           mutate_encoding=mutate_encoding,
                                           cutoff=cutoff)
        arch_dict = search_space.query_arch(mutated, deterministic=deterministic)

        data.append(arch_dict)
        losses.append(arch_dict[loss])
        population.append(len(data) - 1)

        # kill the oldest (or worst) from the population
        if len(population) >= population_size:
            if regularize:
                oldest_index = sorted([i for i in population])[0]
                population.remove(oldest_index)
            else:
                worst_index = sorted([(i, losses[i]) for i in population], key=lambda i: i[1])[-1][0]
                population.remove(worst_index)

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('evolution, query {}, top 5 losses {}'.format(query, top_5_loss))

        query += 1
    return data


def bananas(search_space,
            metann_params,
            num_init=DEFAULT_NUM_INIT,
            k=DEFAULT_K,
            loss=DEFAULT_LOSS,
            total_queries=DEFAULT_TOTAL_QUERIES,
            num_ensemble=5,
            acq_opt_type='mutation',
            num_arches_to_mutate=1,
            max_mutation_rate=1,
            explore_type='its',
            predictor='bananas',
            predictor_encoding='trunc_path',
            cutoff=0,
            mutate_encoding='adj',
            random_encoding='adj',
            deterministic=True,
            verbose=1,
            mutation_tree=None,
            meta_neuralnet=None):
    """
    Bayesian optimization with a neural predictor
    """
    num_init = min(num_init, total_queries // 2)
    data = search_space.generate_random_dataset(num=num_init,
                                                predictor_encoding=predictor_encoding,
                                                random_encoding=random_encoding,
                                                deterministic_loss=deterministic,
                                                cutoff=cutoff)

    query = num_init + k

    while query <= total_queries:

        xtrain = np.array([d['encoding'] for d in data])
        ytrain = np.array([d[loss] for d in data])

        if len(set(d['spec']['string'] for d in data)) >= len(search_space):
            return data

        # get a set of candidate architectures
        candidates = search_space.get_candidates(data,
                                                 acq_opt_type=acq_opt_type,
                                                 predictor_encoding=predictor_encoding,
                                                 mutate_encoding=mutate_encoding,
                                                 num_arches_to_mutate=num_arches_to_mutate,
                                                 max_mutation_rate=max_mutation_rate,
                                                 loss=loss,
                                                 deterministic_loss=deterministic,
                                                 cutoff=cutoff,
                                                 mutation_tree=mutation_tree)

        xcandidates = np.array([c['encoding'] for c in candidates])
        candidate_predictions = []

        # train an ensemble of neural networks
        train_error = 0
        ensemble = []

        for e in range(num_ensemble):

            if predictor == 'bananas':
                if meta_neuralnet is None:
                    meta_neuralnet = MetaNeuralnet()
                net_params = metann_params['ensemble_params'][e]

                train_error += meta_neuralnet.fit(xtrain, ytrain, **net_params)

                # predict the validation loss of the candidate architectures
                candidate_predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))
                tf.compat.v1.reset_default_graph()

            elif predictor == 'gcn':
                """
                Unofficial implementation of BONAS [Shi et al. 2019].
                We used the same BO code as in BANANAS to allow for better comparisons between
                the GCN-based neural predictor and the path-based neural predictor.
                After this code was written, an official implementation of BONAS was released
                at this url: https://github.com/pipilurj/BONAS
                """

                # train GCN and then make predictions on the test data
                if search_space.get_type() == 'nasbench_101':
                    # initial_hidden == num_op_choices + 2
                    initial_hidden = 5
                elif search_space.get_type() == 'nasbench_201':
                    initial_hidden = 7
                else:
                    print('gcn predictor is currently not supported for {}'.format(search_space.get_type()))

                net = NeuralPredictor(initial_hidden=initial_hidden)
                seed = np.random.choice(10000)
                fit(net, xtrain, seed=seed)
                candidate_predictions.append(predict(net, xcandidates))

                # clear memory
                net = None
                gc.collect()

            elif predictor == 'vae':
                """
                Note: vae requires installing additional dependencies
                """

                from vae.train import run_vae
                seed = np.random.choice(10000)
                candidate_predictions.append(run_vae(xtrain, xcandidates, seed=seed))
                gc.collect()

            else:
                print('invalid predictor')

        train_error /= num_ensemble
        if verbose == 2:
            print('query {}, Neural predictor train error: {}'.format(query, train_error))

        # compute the acquisition function for all the candidate architectures
        candidate_indices = acq_fn(candidate_predictions, ytrain=ytrain, explore_type=explore_type)

        # add the k arches with the minimum acquisition function values
        for i in candidate_indices[:k]:
            arch_dict = search_space.query_arch(candidates[i]['spec'],
                                                predictor_encoding=predictor_encoding,
                                                deterministic=deterministic,
                                                cutoff=cutoff)
            data.append(arch_dict)

        tf.keras.backend.clear_session()

        if verbose:
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('{}, query {}, top 5 losses {}'.format(predictor, query, top_5_loss))

        # we just finished performing k queries
        query += k

    return data


def local_search(search_space,
                 num_init=DEFAULT_NUM_INIT,
                 k=DEFAULT_K,
                 loss=DEFAULT_LOSS,
                 random_encoding='adj',
                 mutate_encoding='adj',
                 epochs=0,
                 query_full_nbhd=False,
                 stop_at_minimum=True,
                 total_queries=DEFAULT_TOTAL_QUERIES,
                 deterministic=True,
                 verbose=1):
    """
    local search
    """
    query_dict = {}
    iter_dict = {}
    data = []
    query = 0

    while True:
        # loop over full runs of local search until queries run out

        arch_dicts = []
        while len(arch_dicts) < num_init:
            arch_dict = search_space.query_arch(random_encoding=random_encoding,
                                                deterministic=deterministic,
                                                epochs=epochs)

            if search_space.get_hash(arch_dict['spec']) not in query_dict:
                query_dict[search_space.get_hash(arch_dict['spec'])] = 1
                data.append(arch_dict)
                arch_dicts.append(arch_dict)
                query += 1
                if query >= total_queries:
                    return data

        sorted_arches = sorted([(arch, arch[loss]) for arch in arch_dicts], key=lambda i: i[1])
        arch_dict = sorted_arches[0][0]

        while True:
            # loop over iterations of local search until we hit a local minimum
            iter_dict[search_space.get_hash(arch_dict['spec'])] = 1
            nbhd = search_space.get_nbhd(arch_dict['spec'], mutate_encoding=mutate_encoding)
            improvement = False
            nbhd_dicts = []
            for nbr in nbhd:
                if search_space.get_hash(nbr) not in query_dict:
                    query_dict[search_space.get_hash(nbr)] = 1
                    nbr_dict = search_space.query_arch(nbr,
                                                       deterministic=deterministic,
                                                       epochs=epochs)
                    data.append(nbr_dict)
                    nbhd_dicts.append(nbr_dict)
                    query += 1
                    if query >= total_queries:
                        return data
                    if nbr_dict[loss] < arch_dict[loss]:
                        improvement = True
                        if not query_full_nbhd:
                            arch_dict = nbr_dict
                            break

            if not stop_at_minimum:
                sorted_data = sorted([(arch, arch[loss]) for arch in data], key=lambda i: i[1])
                index = 0
                while search_space.get_hash(sorted_data[index][0]['spec']) in iter_dict:
                    index += 1

                arch_dict = sorted_data[index][0]

            elif not improvement:
                break

            else:
                sorted_nbhd = sorted([(nbr, nbr[loss]) for nbr in nbhd_dicts], key=lambda i: i[1])
                arch_dict = sorted_nbhd[0][0]

        if verbose:
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('local_search, query {}, top 5 losses {}'.format(query, top_5_loss))


def gcn_predictor(search_space,
                  total_queries=DEFAULT_TOTAL_QUERIES,
                  loss=DEFAULT_LOSS,
                  random_encoding='adj',
                  cutoff=0,
                  deterministic=True,
                  verbose=1):
    """
    Implementation of Neural Predictor for Neural Architecture Search [Wen et al. 2020]
    using an unofficial GCN implementation. 
    Note that init=172 is the magic number from their paper, so we use a linear tradeoff
    for init & k which passes through (0,0) and (172, 20)
    """
    k = int(20 / 172 * total_queries)
    num_init = total_queries - k
    acq_size = 50 * total_queries

    # generate the training data
    data = search_space.generate_random_dataset(num=num_init,
                                                deterministic_loss=deterministic,
                                                predictor_encoding='gcn')
    xtrain = [d['encoding'] for d in data]

    # generate the test data
    test_data = search_space.generate_random_dataset(num=acq_size,
                                                     deterministic_loss=deterministic,
                                                     predictor_encoding='gcn',
                                                     train=False)

    test_data = search_space.remove_duplicates(test_data, data)
    xtest = [d['encoding'] for d in test_data]

    if search_space.get_type() == 'nasbench_101':
        # initial_hidden == num_op_choices + 2
        initial_hidden = 5
    elif search_space.get_type() == 'nasbench_201':
        initial_hidden = 7
    else:
        print('gcn predictor is currently not supported for {}'.format(search_space.get_type()))

    # train the neural network   
    net = NeuralPredictor(initial_hidden=initial_hidden)
    seed = np.random.choice(10000)
    fit(net, xtrain, seed=seed)

    # make predictions on the test data
    test_pred = predict(net, xtest)

    # train the k architectures with the best predictions
    sorted_indices = sorted(test_pred)[:k]
    for i in range(k):
        arch_dict = search_space.query_arch(test_data[i]['spec'],
                                            deterministic=deterministic)
        data.append(arch_dict)

    if verbose:
        top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
        print('GCN predictor, top 5 losses: {}'.format(top_5_loss))

    return data


def gp_bayesopt_search(search_space,
                       num_init=DEFAULT_NUM_INIT,
                       k=DEFAULT_K,
                       total_queries=DEFAULT_TOTAL_QUERIES,
                       loss=DEFAULT_LOSS,
                       distance='adj',
                       random_encoding='adj',
                       cutoff=0,
                       deterministic=True,
                       tmpdir='./temp',
                       max_iter=200,
                       mode='single_process',
                       verbose=1,
                       nppred=1000):
    """
    Bayesian optimization with a GP prior
    """

    # set up the path for auxiliary pickle files
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')

    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        return search_space.query_arch(arch,
                                       deterministic=deterministic)[loss]

    # set all the parameters for the various BayesOpt classes
    fhp = Namespace(fhstr='object', namestr='train')
    domp = Namespace(dom_str='list', set_domain_list_auto=True,
                     aux_file_path=aux_file_path,
                     distance=distance)
    modelp = Namespace(kernp=Namespace(ls=3., alpha=1.5, sigma=1e-5),
                       infp=Namespace(niter=num_iterations, nwarmup=500),
                       distance=distance, search_space=search_space.get_type())
    amp = Namespace(am_str='mygpdistmat_ucb', nppred=nppred, modelp=modelp)
    optp = Namespace(opt_str='rand', max_iter=max_iter)
    makerp = Namespace(domp=domp, amp=amp, optp=optp)
    probop = Namespace(niter=num_iterations, fhp=fhp,
                       makerp=makerp, tmpdir=tmpdir, mode=mode)
    data = Namespace()

    # Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init,
                                                     random_encoding=random_encoding,
                                                     deterministic_loss=deterministic)
    data.X = [d['spec'] for d in init_data]
    data.y = np.array([[d[loss]] for d in init_data])

    # initialize aux file
    pairs = [(data.X[i], data.y[i]) for i in range(len(data.y))]
    pairs.sort(key=lambda x: x[1])
    with open(aux_file_path, 'wb') as f:
        pickle.dump(pairs, f)

    # run Bayesian Optimization
    bo = ProBO(fn, search_space, aux_file_path, data, probop, True)
    bo.run_bo(verbose=verbose)

    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in data.X:
        archtuple = search_space.query_arch(arch, deterministic=deterministic)
        results.append(archtuple)

    return results


def pybnn_search(search_space,
                 model_type,
                 num_init=20,
                 k=DEFAULT_K,
                 loss=DEFAULT_LOSS,
                 total_queries=DEFAULT_TOTAL_QUERIES,
                 predictor_encoding='adj',
                 cutoff=0,
                 acq_opt_type='mutation',
                 explore_type='ucb',
                 deterministic=True,
                 verbose=True):
    import torch
    from pybnn import DNGO
    from pybnn.bohamiann import Bohamiann
    from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

    def fn(arch):
        return search_space.query_arch(arch,
                                       deterministic=deterministic)[loss]

    # set up initial data
    data = search_space.generate_random_dataset(num=num_init,
                                                predictor_encoding=predictor_encoding,
                                                cutoff=cutoff,
                                                deterministic_loss=deterministic)

    query = num_init + k

    while query <= total_queries:

        # set up data
        x = np.array([d['encoding'] for d in data])
        y = np.array([d[loss] for d in data])
        scaled_y = np.array([elt / 30 for elt in y])

        # get a set of candidate architectures
        candidates = search_space.get_candidates(data,
                                                 acq_opt_type=acq_opt_type,
                                                 predictor_encoding=predictor_encoding,
                                                 cutoff=cutoff,
                                                 deterministic_loss=deterministic)

        xcandidates = np.array([d['encoding'] for d in candidates])

        # train the model
        if model_type == 'dngo':
            model = DNGO(do_mcmc=False)
            model.train(x, y, do_optimize=True)
        elif model_type == 'bohamiann':
            model = Bohamiann()
            model.train(x, scaled_y, num_steps=10000, num_burn_in_steps=1000, keep_every=50, lr=1e-2)

        predictions, var = model.predict(xcandidates)
        predictions = np.array([pred * 30 for pred in predictions])
        stds = np.sqrt(np.array([v * 30 for v in var]))
        candidate_indices = acq_fn(np.array(predictions), explore_type, stds=stds)

        model = None
        gc.collect()

        # add the k arches with the minimum acquisition function values
        for i in candidate_indices[:k]:
            arch_dict = search_space.query_arch(candidates[i]['spec'],
                                                epochs=0,
                                                predictor_encoding=predictor_encoding,
                                                cutoff=cutoff,
                                                deterministic=deterministic)
            data.append(arch_dict)

        if verbose:
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('dngo, query {}, top 5 val losses: {}'.format(query, top_5_loss))

        query += k

    return data
