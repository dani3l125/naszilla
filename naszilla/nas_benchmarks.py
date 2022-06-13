import numpy as np
import pickle
import sys
import os
import threading
import multiprocessing as mp
import time
import copy
import random
import ctypes

from sklearn_extra.cluster import KMedoids
from torch import load

from nasbench import api
from nas_201_api import NASBench201API as API
import nasbench301 as nb

from naszilla.nas_bench_101.cell_101 import Cell101
from naszilla.nas_bench_201.cell_201 import Cell201
from naszilla.nas_bench_301.cell_301 import Cell301
from naszilla.nas_bench_201.distances import *

default_data_folder = '~/nas_benchmark_datasets/'

def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)

def to_shared_array(arr, ctype):
    shared_array = mp.Array(ctype, arr.size, lock=False)
    return shared_array

class Nasbench:

    def get_cell(self, arch=None):
        return None

    def query_arch(self,
                   arch=None,
                   train=True,
                   predictor_encoding=None,
                   cutoff=0,
                   random_encoding='adj',
                   deterministic=True,
                   epochs=0,
                   random_hash=False,
                   max_edges=None,
                   max_nodes=None):

        arch_dict = {}
        arch_dict['epochs'] = epochs

        if arch is None:
            arch = self.get_cell().random_cell(self.nasbench,
                                               random_encoding=random_encoding,
                                               max_edges=max_edges,
                                               max_nodes=max_nodes,
                                               cutoff=cutoff,
                                               index_hash=self.index_hash)
        arch_dict['spec'] = arch

        if predictor_encoding:
            arch_dict['encoding'] = self.get_cell(arch).encode(predictor_encoding=predictor_encoding,
                                                               nasbench=self.nasbench,
                                                               deterministic=deterministic,
                                                               cutoff=cutoff)

        if train:
            arch_dict['val_loss'] = self.get_cell(arch).get_val_loss(self.nasbench,
                                                                     deterministic=deterministic,
                                                                     dataset=self.dataset)
            arch_dict['test_loss'] = self.get_cell(arch).get_test_loss(self.nasbench,
                                                                       dataset=self.dataset)
            arch_dict['num_params'] = self.get_cell(arch).get_num_params(self.nasbench)
            arch_dict['val_per_param'] = (arch_dict['val_loss'] - 4.8) * (arch_dict['num_params'] ** 0.5) / 100

        return arch_dict

    def mutate_arch(self,
                    arch,
                    mutation_rate=1.0,
                    mutate_encoding='adj',
                    cutoff=0):

        return self.get_cell(arch).mutate(self.nasbench,
                                          mutation_rate=mutation_rate,
                                          mutate_encoding=mutate_encoding,
                                          index_hash=self.index_hash,
                                          cutoff=cutoff)

    def generate_random_dataset(self,
                                num=10,
                                train=True,
                                predictor_encoding=None,
                                random_encoding='adj',
                                deterministic_loss=True,
                                patience_factor=5,
                                allow_isomorphisms=False,
                                cutoff=0,
                                max_edges=None,
                                max_nodes=None):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break

            arch_dict = self.query_arch(train=train,
                                        predictor_encoding=predictor_encoding,
                                        random_encoding=random_encoding,
                                        deterministic=deterministic_loss,
                                        cutoff=cutoff,
                                        max_edges=max_edges,
                                        max_nodes=max_nodes)

            h = self.get_hash(arch_dict['spec'])

            if allow_isomorphisms or h not in dic:
                dic[h] = 1
                data.append(arch_dict)
        return data

    def get_candidates(self,
                       data,
                       num=100,
                       acq_opt_type='mutation',
                       predictor_encoding=None,
                       mutate_encoding='adj',
                       loss='val_loss',
                       allow_isomorphisms=False,
                       patience_factor=5,
                       deterministic_loss=True,
                       num_arches_to_mutate=1,
                       max_mutation_rate=1,
                       train=False,
                       cutoff=0):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """

        candidates = []
        # set up hash map
        dic = {}
        for d in data:
            arch = d['spec']
            h = self.get_hash(arch)
            dic[h] = 1

        if acq_opt_type not in ['mutation', 'mutation_random', 'random']:
            print('{} is not yet implemented as an acquisition type'.format(acq_opt_type))
            raise NotImplementedError()

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest loss
            best_arches = [arch['spec'] for arch in
                           sorted(data, key=lambda i: i[loss])[:num_arches_to_mutate * patience_factor]]

            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime

            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(int(num / num_arches_to_mutate / max_mutation_rate)):
                    for rate in range(1, max_mutation_rate + 1):
                        mutated = self.mutate_arch(arch,
                                                   mutation_rate=rate,
                                                   mutate_encoding=mutate_encoding)
                        arch_dict = self.query_arch(mutated,
                                                    train=train,
                                                    predictor_encoding=predictor_encoding,
                                                    deterministic=deterministic_loss,
                                                    cutoff=cutoff)
                        h = self.get_hash(mutated)

                        if allow_isomorphisms or h not in dic:
                            dic[h] = 1
                            candidates.append(arch_dict)

        if acq_opt_type in ['random', 'mutation_random']:
            # add randomly sampled architectures to the set of candidates
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break

                arch_dict = self.query_arch(train=train,
                                            predictor_encoding=predictor_encoding,
                                            cutoff=cutoff)
                h = self.get_hash(arch_dict['spec'])

                if allow_isomorphisms or h not in dic:
                    dic[h] = 1
                    candidates.append(arch_dict)

        return candidates

    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            dic[self.get_hash(d['spec'])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_hash(candidate['spec']) not in dic:
                dic[self.get_hash(candidate['spec'])] = 1
                unduplicated.append(candidate)
        return unduplicated

    def train_test_split(self, data, train_size,
                         shuffle=True,
                         rm_duplicates=True):
        if shuffle:
            np.random.shuffle(data)
        traindata = data[:train_size]
        testdata = data[train_size:]

        if rm_duplicates:
            self.remove_duplicates(testdata, traindata)
        return traindata, testdata

    def encode_data(self, dicts):
        # input: list of arch dictionary objects
        # output: xtrain (in binary path encoding), ytrain (val loss)

        data = []
        for dic in dicts:
            arch = dic['spec']
            encoding = Arch(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))
        return data

    def get_arch_list(self,
                      aux_file_path,
                      distance=None,
                      iteridx=0,
                      num_top_arches=5,
                      max_edits=20,
                      num_repeats=5,
                      random_encoding='adj',
                      verbose=0):
        # Method used for gp_bayesopt

        # load the list of architectures chosen by bayesopt so far
        base_arch_list = pickle.load(open(aux_file_path, 'rb'))
        top_arches = [archtuple[0] for archtuple in base_arch_list[:num_top_arches]]
        if verbose:
            top_5_loss = [archtuple[1][0] for archtuple in base_arch_list[:min(5, len(base_arch_list))]]
            print('top 5 val losses {}'.format(top_5_loss))

        # perturb the best k architectures    
        dic = {}
        for archtuple in base_arch_list:
            path_indices = self.get_cell(archtuple[0]).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    # perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    perturbation = self.get_cell(arch).mutate(self.nasbench, edits)
                    path_indices = self.get_cell(perturbation).get_path_indices()
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append(perturbation)

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                arch = self.get_cell().random_cell(self.nasbench, random_encoding=random_encoding)
                path_indices = self.get_cell(arch).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(arch)

        return new_arch_list

    # Method used for gp_bayesopt for nasbench
    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        matrix = np.zeros([len(arches_1), len(arches_2)])
        for i, arch_1 in enumerate(arches_1):
            for j, arch_2 in enumerate(arches_2):
                matrix[i][j] = cls.get_cell(arch_1).distance(cls.get_cell(arch_2), dist_type=distance)
        return matrix


class Nasbench101(Nasbench):

    def __init__(self,
                 data_folder=default_data_folder,
                 index_hash_folder='./',
                 mf=False):
        self.mf = mf
        self.dataset = 'cifar10'

        """
        For NAS encodings experiments, some of the path-based encodings currently require a
        hash map from path indices to cell architectuers. We have created a pickle file which
        contains the hash map, located at 
        https://drive.google.com/file/d/1yMRFxT6u3ZyfiWUPhtQ_B9FbuGN3X-Nf/view?usp=sharing
        """
        self.index_hash = None
        index_hash_path = os.path.expanduser(index_hash_folder + 'index_hash.pkl')
        if os.path.isfile(index_hash_path):
            self.index_hash = pickle.load(open(index_hash_path, 'rb'))

        if not self.mf:
            self.nasbench = api.NASBench(os.path.expanduser(data_folder + 'nasbench_only108.tfrecord'))
        else:
            self.nasbench = api.NASBench(os.path.expanduser(data_folder + 'nasbench_full.tfrecord'))

    @classmethod
    def get_cell(cls, arch=None):
        if not arch:
            return Cell101
        else:
            return Cell101(**arch)

    def get_type(self):
        return 'nasbench_101'

    def convert_to_cells(self,
                         arches,
                         predictor_encoding='path',
                         cutoff=0,
                         train=True):
        cells = []
        for arch in arches:
            spec = Cell.convert_to_cell(arch)
            cell = self.query_arch(spec,
                                   predictor_encoding=predictor_encoding,
                                   cutoff=cutoff,
                                   train=train)
            cells.append(cell)
        return cells

    def get_nbhd(self, arch, mutate_encoding='adj'):
        return Cell101(**arch).get_neighborhood(self.nasbench,
                                                mutate_encoding=mutate_encoding,
                                                index_hash=self.index_hash)

    def get_hash(self, arch):
        # return a unique hash of the architecture+fidelity
        return Cell101(**arch).get_path_indices()


class Nasbench201(Nasbench):
    nasbench = None

    def __init__(self,
                 dataset='cifar10',
                 data_folder=default_data_folder,
                 version='1_0',
                 is_debug=False):
        self.search_space = 'nasbench_201'
        self.dataset = dataset
        self.index_hash = None

        if is_debug:
            Nasbench201.nasbench = load(os.path.expanduser(data_folder + 'NAS-Bench-mini.pth'))
        elif version == '1_0':
            Nasbench201.nasbench = API(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_0-e61699.pth'))
        elif version == '1_1':
            Nasbench201.nasbench = API(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_1-096897.pth'))

    def get_type(self):
        return 'nasbench_201'

    @classmethod
    def get_cell(cls, arch=None):
        if not arch:
            return Cell201
        if isinstance(arch, str):
            return Cell201(arch)
        else:
            return Cell201(**arch)

    def get_nbhd(self, arch, mutate_encoding='adj'):
        if isinstance(arch, str):
            return Cell201(arch).get_neighborhood(self.nasbench,
                                                  mutate_encoding=mutate_encoding)
        else:
            return Cell201(**arch).get_neighborhood(self.nasbench,
                                                    mutate_encoding=mutate_encoding)

    def get_hash(self, arch):
        # return a unique hash of the architecture+fidelity
        if isinstance(arch, str):
            return Cell201(arch).get_string()
        else:
            return Cell201(**arch).get_string()


class KNasbench201(Nasbench201):
    _is_updated_distances = False
    _distances = None
    _medoid_indices = None
    old_nasbench = None

    def __init__(self,
                 dataset='cifar10',
                 data_folder=default_data_folder,
                 version='1_0',
                 dim=100,
                 n_threads=16,
                 dist_type='lev'):
        super().__init__(dataset, data_folder, version)
        self.dim = dim
        self.n_threads = n_threads
        self._is_updated_points = False
        self._points = None
        self.dist_type = dist_type
        self.counter = 0

        self._labels = np.zeros(len(self.nasbench))

    def __len__(self):
        return len(self.nasbench)

    @property
    def distances(self):
        start = time.time()
        if KNasbench201._is_updated_distances:
            return KNasbench201._distances

        size = len(self.nasbench)
        KNasbench201._distances = np.zeros((size, size)).astype(np.float32)
        threads = []
        cells_l = []
        for i in range(len(self.nasbench)):
            cells_l.append(
                self.get_cell(KNasbench201.nasbench.meta_archs[KNasbench201.nasbench.evaluated_indexes.index(i)],  init=True))

        if self.dist_type == 'real':
            values = np.zeros(size)
            for i in range(size):
                values[i] = self.nasbench.query_meta_info_by_index(i).get_metrics(self.dataset, 'train')['accuracy']

            d_row = np.tile(values, (size, 1))
            d_col = d_row.T
            KNasbench201._distances = (d_row - d_col) ** 2
        # elif self.dist_type == 'lev':
        #     KNasbench201._distances = np.load('lev_dist.npy')

        else:
            def calc_dist(buf, module):
                distances = to_numpy_array(buf, (size, size))
                for i in range(len(self.nasbench)):
                    if i % self.n_threads == module:
                        self.counter += 1
                        for j in range(i, len(self.nasbench)):
                            if self.dist_type == 'real':
                                distances[i, j] = distances[j, i] = \
                                self.nasbench.query_meta_info_by_index(
                                    self.nasbench.query_index_by_arch(cells_l[i].string)).get_metrics(self.dataset,
                                                                                                      'train')['accuracy'] - \
                                self.nasbench.query_meta_info_by_index(
                                    self.nasbench.query_index_by_arch(cells_l[j].string)).get_metrics(self.dataset,
                                                                                                      'train')['accuracy']
                            else:
                                distances[i, j] = distances[j, i] = cells_l[i].distance(
                                    cells_l[j], self.dist_type, self)
                            # print(KNasbench201._distances[i, j])

            shared_buf = to_shared_array(KNasbench201._distances, ctypes.c_float)
            for t_idx in range(self.n_threads):
                t = mp.Process(target=calc_dist, args=(shared_buf, t_idx,))
                t.start()
                threads.append(t)
            time.sleep(10)
            for t in threads:
                t.join()
            KNasbench201._distances = to_numpy_array(shared_buf, (size, size))
        np.save('lev_dist.npy', KNasbench201._distances)

        KNasbench201._is_updated_distances = True
        return KNasbench201._distances

    @property
    def points(self):
        if self._is_updated_points:
            return self._points
        # TODO: check tile dimentions order
        m_row = np.tile(self.distances[0] ** 2, (1, self.distances.shape[0]))
        m_col = np.tile(self.distances.T[0] ** 2, (1, self.distances.shape[0])).T
        M = (self.distances ** 2 + m_row + m_col) / 2

        U, S, V = np.linalg.svd(M)
        X = U @ np.sqrt(S)

        self._is_updated_points = True
        self._points = X[:, :self.dim + 1]
        return self._points

    def get_type(self):
        return 'knasbench_201'

    def copy_bench(self):
        KNasbench201.old_nasbench = copy.deepcopy(KNasbench201.nasbench)

    def remove_by_indices(self, indices):
        for idx in indices:
            # a_idx = KNasbench201.nasbench.evaluated_indexes.index(idx)
            arch_str = KNasbench201.nasbench.arch2infos_full[idx].arch_str
            del KNasbench201.nasbench.arch2infos_full[idx]
            del KNasbench201.nasbench.arch2infos_less[idx]
            del KNasbench201.nasbench.archstr2index[arch_str]
            KNasbench201.nasbench.evaluated_indexes.remove(idx)
            KNasbench201.nasbench.meta_archs.remove(arch_str)

    def parallel_remove(self, remove_indices):
        threads = []
        chunk_size = int(len(remove_indices) / (self.n_threads - 1))
        first_idx = 0
        last_idx = chunk_size
        for i in range(self.n_threads - 1):
            t = mp.Process(target=self.remove_by_indices, args=([remove_indices[first_idx:last_idx]]))
            t.start()
            threads.append(t)
            first_idx += chunk_size
            last_idx += chunk_size
        if last_idx >= len(remove_indices) - 1:
            t = mp.Process(target=self.remove_by_indices, args=([remove_indices[last_idx:]]))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def prune(self, k=20):
        start = time.time()
        # TODO: debug deepcopy paralelizing
        # copy_thread = Process(target=KNasbench201.copy_bench)
        # copy_thread.start()
        KNasbench201.old_nasbench = copy.deepcopy(KNasbench201.nasbench)
        kmedoids = KMedoids(n_clusters=k, metric='precomputed').fit(  # Take distances from current cluster
            self.distances[KNasbench201.nasbench.evaluated_indexes][:, KNasbench201.nasbench.evaluated_indexes])
        self._labels = kmedoids.labels_
        KNasbench201._medoid_indices = kmedoids.medoid_indices_
        # copy_thread.join()
        remove_indices = list(set(range(len(KNasbench201.nasbench))) - set(kmedoids.medoid_indices_))
        self.parallel_remove(remove_indices)

        print(f'#############\n\nSpace updated to center. total time: {time.time() - start}\n\n#############')

    def choose_clusters(self, data, m):
        start = time.time()
        KNasbench201.nasbench = KNasbench201.old_nasbench
        best_dicts = sorted(data, key=lambda x:x['test_loss'])[:m]  # there is val_loss also.
        remove_indices = set(KNasbench201.nasbench.evaluated_indexes)
        for best_dict in best_dicts:
            best_str = best_dict['spec']['string']
            cluster_idx = self._labels[KNasbench201.nasbench.archstr2index[best_str]]
            cluster_elements_indexes_list = np.where(self._labels == cluster_idx)[0]  # Indexes in list only!
            real_indexes = np.array(KNasbench201.nasbench.evaluated_indexes)[cluster_elements_indexes_list]
            remove_indices = remove_indices - set(real_indexes)
        remove_indices = list(remove_indices)
        self.parallel_remove(remove_indices)
        print(f'#############\n\nSpace updated to cluster. total time: {time.time() - start}\n\n#############')


    @classmethod
    def get_cell(cls, arch=None, init=False):
        if not arch:
            return Cell201

        if not cls.nasbench is None and not init:
            # if not cls._is_updated_distances:
            #     raise Exception('Distances are not updated properly.')

            # Choose nearest sample in new set every time
            idx = cls.old_nasbench.archstr2index[arch] if isinstance(arch, str) else cls.old_nasbench.archstr2index[
                arch['string']]
            candidates = cls._distances[idx][np.array(cls.nasbench.evaluated_indexes).astype(int)]
            nearest_idx = np.argmin(candidates)  # this is the index in evaluated indexes list
            arch = cls.nasbench.meta_archs[nearest_idx]

        if isinstance(arch, str):
            return Cell201(arch)
        else:
            return Cell201(**arch)


class Nasbench301(Nasbench):

    def __init__(self,
                 data_folder=default_data_folder
                 ):
        self.dataset = 'cifar10'
        self.search_space = 'nasbench_301'
        ensemble_dir_performance = os.path.expanduser(data_folder + 'nb_models/xgb_v0.9')
        performance_model = nb.load_ensemble(ensemble_dir_performance)
        ensemble_dir_runtime = os.path.expanduser(data_folder + 'nb_models/lgb_runtime_v0.9')
        runtime_model = nb.load_ensemble(ensemble_dir_runtime)
        self.nasbench = [performance_model, runtime_model]
        self.index_hash = None

    def get_type(self):
        return 'nasbench_301'

    @classmethod
    def get_cell(cls, arch=None):
        if not arch:
            return Cell301
        else:
            return Cell301(**arch)

    def get_nbhd(self, arch, mutate_encoding='adj'):
        return Cell301(**arch).get_neighborhood(self.nasbench,
                                                mutate_encoding=mutate_encoding)

    def get_hash(self, arch):
        # return a unique hash of the architecture+fidelity
        return Cell301(**arch).serialize()
