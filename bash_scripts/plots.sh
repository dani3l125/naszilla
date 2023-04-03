#!/bin/bash

TRIALS=1

let NTHREADS=$TRIALS*4+1
for i in 3
do
#  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10 --study 1 --first 1 --last 10
##
#  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100 --study 1 --first 1 --last 10
#
  /home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset ImageNet16-120 --study 1 --first 10 --last 300

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

#/home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#--algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/1.yaml\
#  --dataset cifar10 --study 0 --first 10 --last 300 --exp_name i10_m2_ciss1_k100_r2_path
#
#/home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#--algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/1.yaml\
#  --dataset cifar100 --study 0 --first 10 --last 300 --exp_name i10_m2_ciss1_k100_r2_path
#
#/home/daniel/miniconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/plot.py --search_space nasbench_201 \
#--algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/1.yaml\
#  --dataset ImageNet16-120 --study 0 --first 10 --last 300 --exp_name i10_m2_ciss1_k100_r2_path