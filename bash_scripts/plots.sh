#!/bin/bash

TRIALS=1

let NTHREADS=$TRIALS*4+1
for i in 2 5 6
do
#  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10 --study 1 --first 1 --last 10
##
#  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100 --study 1 --first 1 --last 10

#  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120 --study 1 --first 10 --last 300

  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar10 --study 0 --first 10 --last 300
##
  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar100 --study 0 --first 10 --last 300

  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset ImageNet16-120 --study 0 --first 10 --last 300
done