#!/bin/bash

NTHREADS=1
TRIALS=1
datasets=('ImageNet16-120')
algos=('evolution' 'bananas' 'local_search' 'random')
studies=(0)
cfgs=('/home/daniel/naszilla/naszilla/config_files/3.yaml')

for dataset in ${datasets[@]}; do
  for cfg in ${cfgs[@]}; do
    for study in ${studies[@]}; do
      for algo in ${algos[@]}; do
        /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
        --algo_params ${algo} --queries 300 --trials ${TRIALS} --k_alg 1 --cfg ${cfg}\
        --dataset ${dataset}
      done
    done
  done
done