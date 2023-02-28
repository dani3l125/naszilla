#!/bin/bash

TRIALS=1

for i in 1 2
do
  screen -L -Logfile cifar10_exp$i -S cifar10_exp$i -d -m bash -c\
    "/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
  --algo_params bananas --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar10"

  screen -L -Logfile cifar100_exp$i -S cifar100_exp$i -d -m bash -c\
    "/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
  --algo_params bananas --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar100"

  screen -L -Logfile imagenet_exp$i -S imagenet_exp$i -d -m bash -c\
    "/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
  --algo_params bananas --queries 300 --trials $TRIALS --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset ImageNet16-120"

#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
done

