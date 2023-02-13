#!/bin/bash

TRIALS=1

let NTHREADS=$TRIALS*4+1
for i in 1 2 3 4 5
do
  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar10
#
  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar100

  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset ImageNet16-120
done