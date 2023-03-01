import numpy as np
import pickle
import os
import multiprocessing as mp
import time
import copy
import ctypes

from sklearn_extra.cluster import KMedoids
from torch import load
from torch.cuda import is_available
if is_available():
    import cupy as cp
else:
    import numpy as cp


from nasbench import api
from nas_201_api import NASBench201API as API
import nasbench301 as nb

from naszilla.nas_bench_101.cell_101 import Cell101
from naszilla.nas_bench_201.cell_201 import Cell201
from naszilla.nas_bench_301.cell_301 import Cell301
from naszilla.nas_bench_201.distances import *
from naszilla.coresets.k_means_coreset_via_robust_median import knas_coreset

default_data_folder = '~/nas_benchmark_datasets/'
CUDA = 0


def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)


def to_shared_array(arr, ctype):
    shared_array = mp.Array(ctype, arr.size, lock=False)
    return shared_array


class MutationTree:
    def __init__(self, archs_list):
        self.prefix_dict = {'': set()}
        for arch in archs_list:
            path = Cell201(arch).get_op_list()
            key = ''
            for node in path:
                self.prefix_dict[key].add(node)
                key += node
                if key not in self.prefix_dict.keys():
                    self.prefix_dict[key] = set()

    def ops(self, path=[]):
        path = ''.join(path)
        if not path in self.prefix_dict.keys():
            return []
        return list(self.prefix_dict[path])


def get_intersections(r1, r2, m1, m2):
    return []


def get_intersections_disjoint(r1, r2, m1, m2):
    step = np.linalg.norm(m2 - m1)
    i1 = m1 + r1 * step
    i2 = m1 - r1 * step
    i = i1 if ((i1 - m1) ** 2).mean() < ((i2 - m1) ** 2).mean() else i2
    i1 = m2 + r1 * step
    i2 = m2 - r1 * step
    return [i, i1 if ((i1 - m2) ** 2).mean() < ((i2 - m2) ** 2).mean() else i2]


def get_intersections_included(r1, r2, m1, m2):
    return []


class Nasbench:

    def is_knas(self):
        return False

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
                    cutoff=0,
                    mutation_tree=None):

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
                                allow_isomorphisms=True,
                                cutoff=0,
                                max_edges=None,
                                max_nodes=None):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        num = min(num, len(self.nasbench))
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

    def generate_complete_dataset(self):
        raise NotImplementedError

    def get_candidates(self,
                       data,
                       num=100,
                       acq_opt_type='mutation',
                       predictor_encoding=None,
                       mutate_encoding='adj',
                       loss='val_loss',
                       allow_isomorphisms=False,
                       # TODO: handle isomorphisms
                       patience_factor=5,
                       deterministic_loss=True,
                       num_arches_to_mutate=1,
                       max_mutation_rate=1,
                       train=False,
                       cutoff=0,
                       mutation_tree=None):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """

        candidates = []
        counter = 0
        # set up hash map
        while len(candidates) == 0:
            counter += 1
            if counter >= 2:
                print('Candidates loop reached decond iteration')
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
                                                       mutate_encoding=mutate_encoding,
                                                       mutation_tree=mutation_tree)
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

    def __init__(self,
                 dataset='cifar10',
                 data_folder=default_data_folder,
                 version='1_0',
                 is_debug=False):
        self.search_space = 'nasbench_201'
        self.dataset = dataset
        self.index_hash = None

        print(f'\t\t\t\nis debug  =  {is_debug}')

        if is_debug:
            self.nasbench = load(os.path.expanduser(data_folder + 'NAS-Bench-mini.pth'))
        elif version == '1_0':
            self.nasbench = API(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_0-e61699.pth'))
        elif version == '1_1':
            self.nasbench = API(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_1-096897.pth'))

    def __len__(self):
        return len(self.nasbench)

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

    def get_nbhd(self, arch, mutate_encoding='adj', arch_list=None):
        if isinstance(arch, str):
            return Cell201(arch).get_neighborhood(mutate_encoding=mutate_encoding,
                                                  arch_list=arch_list)
        else:
            return Cell201(**arch).get_neighborhood(mutate_encoding=mutate_encoding,
                                                  arch_list=arch_list)

    def get_hash(self, arch):
        # return a unique hash of the architecture+fidelity
        if isinstance(arch, str):
            return Cell201(arch).get_string()
        else:
            return Cell201(**arch).get_string()

    def generate_complete_dataset(self):
        data = []
        # if len(data) == 0:
        #     print('1')
        for arch in self.nasbench.meta_archs:
            data.append(self.query_arch(arch))
        return data


class KNasbench201(Nasbench201):

    def __init__(self,
                 dataset='cifar10',
                 data_folder=default_data_folder,
                 version='1_0',
                 dim=15,
                 n_threads=16,
                 dist_type='lev',
                 compression_method='k_medoids',
                 compression_args=None,
                 points_alg='evd',
                 is_debug=False
                 ):
        super().__init__(dataset, data_folder, version, is_debug=is_debug)
        self._is_updated_distances = False
        self._distances = None
        self.coreset_indexes = None
        self.old_nasbench = None
        self._points = None

        print(f'\t\t\t\nis debug  =  {is_debug}')

        # if is_debug:
        #     self.nasbench = load(os.path.expanduser(data_folder + 'NAS-Bench-mini.pth'))
        # elif version == '1_0':
        #     self.nasbench = API(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_0-e61699.pth'))
        # elif version == '1_1':
        #     self.nasbench = API(os.path.expanduser(data_folder + 'NAS-Bench-201-v1_1-096897.pth'))
        self.sizes_list = []
        self.dim = dim
        self.n_threads = n_threads
        self._is_updated_points = False
        self.dist_type = dist_type
        self.counter = 0
        self.ratio = -1
        self.cluster_sizes = None
        self.points_alg = points_alg
        self.compression_method = compression_method
        self.compression_kwargs = compression_args
        if isinstance(self.compression_kwargs['k'], list):
            self.k_for_coreset = self.compression_kwargs['k']
        else:
            self.k_for_coreset = np.ones([50]) * self.compression_kwargs['k']

        self.labels = np.zeros(len(self.nasbench))

    def __len__(self):
        return len(self.nasbench)

    def is_knas(self):
        return True

    @property
    def distances(self):
        if self._is_updated_distances:
            return self._distances

        if os.path.isfile(f'distances/{self.dist_type}_dist.npy'):
            print('Using pre-computed distances...')
            self._distances = np.load(f'distances/{self.dist_type}_dist.npy')
            return self._distances

        size = len(self.nasbench)
        self._distances = np.zeros((size, size)).astype(np.float32)
        threads = []
        cells_l = []
        for i in range(len(self.nasbench)):
            cells_l.append(
                self.get_cell(self.nasbench.meta_archs[self.nasbench.evaluated_indexes.index(i)],
                              init=True))

        if self.dist_type == 'real':
            values = np.zeros(size)
            for i in range(size):
                values[i] = self.nasbench.query_meta_info_by_index(i).get_metrics(self.dataset, 'train')['accuracy']

            d_row = np.tile(values, (size, 1))
            d_col = d_row.T
            self._distances = (d_row - d_col) ** 2
        # elif self.dist_type == 'lev':
        #     self._distances = np.load('lev_dist.npy')

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
                                                                                                          'train')[
                                        'accuracy'] - \
                                    self.nasbench.query_meta_info_by_index(
                                        self.nasbench.query_index_by_arch(cells_l[j].string)).get_metrics(self.dataset,
                                                                                                          'train')[
                                        'accuracy']
                            else:
                                distances[i, j] = distances[j, i] = cells_l[i].distance(
                                    cells_l[j], self.dist_type, self)
                            # print(self._distances[i, j])

            shared_buf = to_shared_array(self._distances, ctypes.c_float)
            for t_idx in range(self.n_threads):
                t = mp.Process(target=calc_dist, args=(shared_buf, t_idx,))
                t.start()
                threads.append(t)
            time.sleep(10)
            for t in threads:
                t.join()
            self._distances = to_numpy_array(shared_buf, (size, size))


        self._distances /= np.max(np.abs(self._distances))
        np.save('dist.npy', self._distances)
        self._is_updated_distances = True
        return self._distances

    @property
    def points(self):
        '''
            Returns a matrix where each row is a point approximation
        '''
        if self._is_updated_points:
            return self._points

        start = time.time()

        dist_matrix = np.array(self.distances[np.array(self.nasbench.evaluated_indexes)].T[
            np.array(self.nasbench.evaluated_indexes)], dtype=np.float16)

        if self.points_alg == 'evd':
            m_row = np.tile(dist_matrix[0] ** 2, (dist_matrix.shape[0], 1))
            m_col = np.tile(dist_matrix.T[0] ** 2, (dist_matrix.shape[0], 1)).T
            M = (dist_matrix ** 2 + m_row + m_col) / 2
            w, v = np.linalg.eigh(M.astype(np.float32))
            w = w.astype(np.float16)
            v = v.astype(np.float16)
            sign = np.tile(np.sign(w), (w.shape[0], 1))
            w_sqrt = np.tile(np.sqrt(np.abs(w.real)), (w.shape[0], 1))
            X = v.real * w_sqrt * sign

            self.dim = min(self.dim, M.shape[0])
            ind = np.argpartition(np.abs(w), -self.dim)[-self.dim:]
            self._points = X.T[ind].T

        elif self.points_alg == 'icba':
            if self.dim > 2:
                raise NotImplementedError('ICBA algorithm only implemented for d=2')
            with cp.cuda.Device(CUDA):
                point2 = cp.zeros((1, self.dim))
                point2[0][0] = dist_matrix[0][1]
                points = [cp.zeros((1, self.dim), dtype=cp.float16), point2]  # TODO start with 3 points
                d_centers = cp.zeros((1, 1))  # Insert first and second points
                for m in range(2, len(self.nasbench)):
                    start_i = time.time()
                    # Radiuses vector
                    R = cp.array(dist_matrix[m, :m])
                    # Points matrix
                    centers = cp.concatenate(points, dtype=cp.float16)
                    centers_tile = cp.tile(cp.expand_dims(centers, 0), len(points)).reshape(len(points), len(points), self.dim)
                    # line representative vectors
                    line_vecs = centers_tile - cp.transpose(centers_tile,axes=(1, 0, 2))
                    vecs_sums = cp.sum(line_vecs, axis=2)
                    vecs_sums = cp.concatenate((cp.expand_dims(vecs_sums, -1), cp.expand_dims(vecs_sums, -1)), axis=2, dtype=cp.float16)
                    line_vecs = line_vecs / vecs_sums
                    # Calculate points on circles for disjoint intersections projection
                    R_tile = cp.tile(cp.expand_dims(R, 0).T, R.shape[0]).T
                    R_tile = cp.concatenate((cp.expand_dims(R_tile, -1), cp.expand_dims(R_tile, -1)), axis=2, dtype=cp.float16)
                    steps = R_tile * line_vecs
                    intersections1 = cp.reshape(centers_tile - steps, (-1, self.dim))
                    intersections2 = cp.reshape(centers_tile + steps, (-1, self.dim))
                    disjoint_intersections = cp.concatenate((intersections1, intersections2), dtype=cp.float16)

                    # Real intersections based on http://paulbourke.net/geometry/circlesphere/
                    R_tile = cp.tile(cp.expand_dims(R, 0).T, R.shape[0])
                    R_squared_distances =  R_tile ** 2 - R_tile.T **2
                    centers_squared_distances = cp.sum(line_vecs ** 2, axis=2)
                    A = (R_squared_distances + centers_squared_distances) / (2 * cp.sqrt(centers_squared_distances))
                    P2 = centers_tile + cp.concatenate((cp.expand_dims(A, -1), cp.expand_dims(A, -1)), axis=2, dtype=cp.float16) * line_vecs
                    H = R_tile ** 2 - A ** 2
                    H = cp.concatenate((cp.expand_dims(H, -1), cp.expand_dims(H, -1)), axis=2, dtype=cp.float16)
                    intersections1 = cp.reshape(P2 + H * line_vecs, (-1, self.dim))
                    intersections2 = cp.reshape(P2 - H * line_vecs, (-1, self.dim))

                    intersections = cp.concatenate((intersections1, intersections2, disjoint_intersections), dtype=cp.float16)
                    intersections = intersections[~cp.isnan(intersections), ~cp.isnan(intersections)]
                    intersections_losses = cp.nan_to_num(cp.sum(cp.abs(cp.linalg.norm(intersections[:, None, :] - centers[None, :, :] ,axis = -1),cp.tile(cp.expand_dims(R, 0),intersections.shape[0]).reshape(-1, m)),axis=1), nan=cp.inf)

                    points.append(cp.expand_dims(intersections[cp.argmin(intersections_losses)], 0))
                    print(f'ICBA iteration={m}, distance={cp.min(intersections_losses)}, time={time.time() -start_i}')

            self._points = np.concatenate(points).get()

        else:
            raise NotImplementedError('Invalid points computation algorithm')

        print(f'Points computed. time: {time.time() - start}')
        return self._points

    def get_best_arch_loss(self):
        best_loss = 100
        best_idx = -1
        for arch_idx in self.nasbench.evaluated_indexes:
            dataset = self.dataset if self.dataset != 'cifar10' else 'cifar10-valid'
            x = self.nasbench.query_by_index(arch_idx).get_metrics(dataset, 'x-valid')['loss']
            if x < best_loss:
                best_loss = x
                best_idx = arch_idx

        return best_idx, best_loss

    def get_type(self):
        return 'knasbench_201'

    def copy_bench(self):
        self.old_nasbench = copy.deepcopy(self.nasbench)

    def remove_by_indices(self, indices):
        # print(f'Indexes to remove:{indices}')
        for idx in indices:
            # a_idx = self.nasbench.evaluated_indexes.index(idx)
            arch_str = self.nasbench.arch2infos_full[idx].arch_str
            del self.nasbench.arch2infos_full[idx]
            del self.nasbench.arch2infos_less[idx]
            del self.nasbench.archstr2index[arch_str]
            self.nasbench.evaluated_indexes.remove(idx)
            self.nasbench.meta_archs.remove(arch_str)

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

    def prune(self, iteration, k=20):
        start = time.time()
        # TODO: debug deepcopy paralelizing
        # copy_thread = Process(target=KNasbench201.copy_bench)
        # copy_thread.start()
        self.old_nasbench = copy.deepcopy(self.nasbench)
        print(f'Compression: {self.compression_method}')

        self.compression_kwargs['greedy'] = (self.compression_method == 'k_centers_greedy')

        if self.compression_method == 'uniform':
            self.coreset_indexes = np.random.choice(len(self.nasbench.evaluated_indexes),k)
            points2coreset_dist_mat = self.distances[self.nasbench.evaluated_indexes][
                                      :, self.nasbench.evaluated_indexes][
                                      :, self.coreset_indexes]
            self.labels = np.argmin(points2coreset_dist_mat, axis=1)

        elif self.compression_method == 'k_medoids':
            kmedoids = KMedoids(n_clusters=k, metric='precomputed').fit(  # Take distances from current cluster
                self.distances[self.nasbench.evaluated_indexes][:, self.nasbench.evaluated_indexes])
            self.labels = kmedoids.labels_
            self.coreset_indexes = kmedoids.medoid_indices_

        elif self.compression_method == 'k_centers_coreset_geometric':
            if self.compression_kwargs['k_ratio']:
                self.compression_kwargs['k'] = int(len(self.nasbench) * self.compression_kwargs['k_ratio'])
            else:
                self.compression_kwargs['k'] = self.k_for_coreset[iteration]
            self.compression_kwargs['r'] = 2
            self.compression_kwargs['sum_to_max'] = 1
            self.coreset_indexes, self.labels = knas_coreset(
                self.points, None, **self.compression_kwargs)
            k = self.coreset_indexes.shape[0]


        elif self.compression_method in ['k_means_coreset', 'k_medians_coreset', 'k_centers_coreset', 'k_centers_greedy']:
            if self.compression_kwargs['k_ratio']:
                self.compression_kwargs['k'] = int(len(self.nasbench) * self.compression_kwargs['k_ratio'])
            else:
                self.compression_kwargs['k'] = self.k_for_coreset[iteration]
                self.compression_kwargs['r'] = 1 if self.compression_method == 'k_medians_coreset' else 2
                self.compression_kwargs['sum_to_max'] = 1 if self.compression_method == 'k_centers_coreset' else 0
                self.coreset_indexes, self.labels = knas_coreset(
                self.points, self.distances[self.nasbench.evaluated_indexes][:, self.nasbench.evaluated_indexes],
                **self.compression_kwargs)
            k = self.coreset_indexes.shape[0]

        else:
            raise NotImplementedError('Invalid compression type')
        # self.parallel_remove(remove_indices)
        # self.mutation_tree = MutationTree(self.nasbench.meta_archs)

        self.labels[self.coreset_indexes] = np.arange(k)
        self.cluster_sizes = np.bincount(self.labels)
        # copy_thread.join()
        remove_indices = list(set(self.nasbench.evaluated_indexes) -
                              set(np.array(self.nasbench.evaluated_indexes)[
                                      self.coreset_indexes]))
        self.remove_by_indices(remove_indices)
        print(f'\nSpace updated to centers.\ntime: {time.time() - start}\nsize:{len(self.nasbench)}\n')
        self.ratio = k / len(self.nasbench)
        self.sizes_list.append(self.coreset_indexes.size)
        return k

    def cluster_by_arch(self, arch):
        if isinstance(arch, dict):
            # _labels indexes are incorrext
            return self.labels[
                self.nasbench.evaluated_indexes.index(self.nasbench.archstr2index[arch['string']])]
        return self.labels[self.nasbench.evaluated_indexes.index(self.nasbench.archstr2index[arch])]

    def choose_clusters(self, data, m):
        start = time.time()
        # When m is 0, choose it automatically to save query-data ratio
        best_dicts = sorted(data, key=lambda x: x['val_loss'])
        self.nasbench = self.old_nasbench
        # Remove duplicates
        best_dicts = list({best_dict['spec']['string']:best_dict for best_dict in best_dicts}.values())
        best_dicts = best_dicts[:m]
        remove_indices = set(self.nasbench.evaluated_indexes)
        for best_dict in best_dicts:
            cluster_idx = self.cluster_by_arch(best_dict['spec'])
            cluster_elements_indexes_list = np.where(self.labels == cluster_idx)[0]  # Indexes in list only!
            real_indexes = np.array(self.nasbench.evaluated_indexes)[cluster_elements_indexes_list]
            remove_indices = remove_indices - set(real_indexes)
        remove_indices = list(remove_indices)
        self.remove_by_indices(remove_indices)
        print(f'\nSpace updated to clusters.\ntime: {time.time() - start}\nsize:{len(self.nasbench)}')
        self._is_updated_points = False

    def get_sizes_list(self):
        return self.sizes_list

    @classmethod
    def get_cell(cls, arch=None, init=False):
        if arch is None:
            return Cell201

        # if not cls.nasbench is None and not init:
        #     # if not cls._is_updated_distances:
        #     #     raise Exception('Distances are not updated properly.')
        #
        #     # Choose nearest sample in new set every time
        #     idx = cls.old_nasbench.archstr2index[arch] if isinstance(arch, str) else cls.old_nasbench.archstr2index[
        #         arch['string']]
        #     candidates = cls._distances[idx][np.array(cls.nasbench.evaluated_indexes).astype(int)]
        #     nearest_idx = np.argmin(candidates)  # this is the index in evaluated indexes list
        #     arch = cls.nasbench.meta_archs[nearest_idx]

        if isinstance(arch, str):
            return Cell201(arch)
        else:
            return Cell201(**arch)

    def mutate_arch(self,
                    arch,
                    mutation_rate=1.0,
                    mutate_encoding='adj',
                    cutoff=0,
                    mutation_tree=None):

        return self.get_cell(arch).mutate(self.nasbench,
                                          mutation_rate=mutation_rate,
                                          mutate_encoding=mutate_encoding,
                                          index_hash=self.index_hash,
                                          cutoff=cutoff,
                                          mutation_tree=mutation_tree)

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
                    perturbation = self.get_cell(arch).mutate(self.nasbench, edits, mutation_tree=self.mutation_tree)
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
