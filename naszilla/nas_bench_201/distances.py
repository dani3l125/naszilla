from functools import lru_cache
from Levenshtein import distance as levenshtein
import numpy as np

OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)
OP_SPOTS = 6
LONGEST_PATH_LENGTH = 3


def adj_distance(cell_1, cell_2):
    """
    compute the distance between two architectures
    by comparing their adjacency matrices and op lists
    (edit distance)
    """
    graph_dist = np.sum(np.array(cell_1.get_matrix()) != np.array(cell_2.get_matrix()))
    ops_dist = np.sum(np.array(cell_1.get_ops()) != np.array(cell_2.get_ops()))
    return graph_dist + ops_dist

def jdpath_distance(cell_1, cell_2, cutoff=None):
    """
    compute the distance between two architectures
    by comparing their path encodings
    """
    if cutoff:
        encode_1 = np.array(cell_1.encode('trunc_path', cutoff=cutoff))
        encode_2 = np.array(cell_2.encode('trunc_path', cutoff=cutoff))
    else:
        encode_1 = np.array(cell_1.encode('path', cutoff=cutoff))
        encode_2 = np.array(cell_2.encode('path', cutoff=cutoff))
    a = np.sum(encode_1 and encode_2)
    b = np.sum(encode_1 and not encode_2)
    c = np.sum(not encode_1 and encode_2)
    return (b+c)/(a+b+c)

def path_distance(cell_1, cell_2, cutoff=None):
    """ 
    compute the distance between two architectures
    by comparing their path encodings
    """
    if cutoff:
        return np.sum(np.array(cell_1.encode('trunc_path', cutoff=cutoff) != np.array(cell_2.encode('trunc_path', cutoff=cutoff))))
    else:
        return np.sum(np.array(cell_1.encode('path') != np.array(cell_2.encode('path'))))

def adj_distance(cell_1, cell_2):

    cell_1_ops = cell_1.get_op_list()
    cell_2_ops = cell_2.get_op_list()
    return np.sum([1 for i in range(len(cell_1_ops)) if cell_1_ops[i] != cell_2_ops[i]])

def nasbot_distance(cell_1, cell_2):
    # distance based on optimal transport between row sums, column sums, and ops

    cell_1_ops = cell_1.get_op_list()
    cell_2_ops = cell_2.get_op_list()

    cell_1_counts = [cell_1_ops.count(op) for op in OPS]
    cell_2_counts = [cell_2_ops.count(op) for op in OPS]
    ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))

    return ops_dist + adj_distance(cell_1, cell_2)

def lev_distance(cell_1, cell_2):
    return levenshtein(cell_1.string, cell_2.string)

    # a = cell_1.string
    # b = cell_2.string
    # @lru_cache(None)  # for memorization
    # def min_dist(s1, s2):
    #
    #     if s1 == len(a) or s2 == len(b):
    #         return len(a) - s1 + len(b) - s2
    #
    #     # no change required
    #     if a[s1] == b[s2]:
    #         return min_dist(s1 + 1, s2 + 1)
    #
    #     return 1 + min(
    #         min_dist(s1, s2 + 1),      # insert character
    #         min_dist(s1 + 1, s2),      # delete character
    #         min_dist(s1 + 1, s2 + 1),  # replace character
    #     )
    #
    # return min_dist(0, 0)


def real_distance(cell_1, cell_2, nasbench):
    return nasbench.nasbench.query_meta_info_by_index(nasbench.nasbench.query_index_by_arch(cell_1.string)).get_metrics(nasbench.dataset, 'train')['accuracy'] - \
    nasbench.nasbench.query_meta_info_by_index(nasbench.nasbench.query_index_by_arch(cell_2.string)).get_metrics(nasbench.dataset, 'train')['accuracy']
