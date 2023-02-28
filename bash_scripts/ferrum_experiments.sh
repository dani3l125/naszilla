#!/bin/bash

NTHREADS=11
TRIALS=10
datasets=('cifar10' 'cifar100' 'ImageNet16-120')
algos=('random' 'local_search' 'evolution', 'bananas')
studies=(1)
cfgs=('/home/daniel/naszilla/naszilla/config_files/1.yaml')

for dataset in ${datasets[@]}; do
  for cfg in ${cfgs[@]}; do
    for study in ${studies[@]}; do
      for algo in ${algos[@]}; do
        /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
        --algo_params ${algo} --queries 300 --trials ${$TRIALS} --k_alg 1 --cfg ${cfg}\
        --dataset ${dataset}
      done
    done
  done
done