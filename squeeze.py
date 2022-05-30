import torch

from nas_201_api import NASBench201API as API

nasbench = torch.load('NAS-Bench-mini.pth')
for idx in range(150, len(nasbench)):
    arch_str = nasbench.arch2infos_full[idx].arch_str
    del nasbench.arch2infos_full[idx]
    del nasbench.arch2infos_less[idx]
    del nasbench.archstr2index[arch_str]
    nasbench.evaluated_indexes.remove(idx)
    nasbench.meta_archs.remove(arch_str)

torch.save(nasbench, 'NAS-Bench-mini.pth')
