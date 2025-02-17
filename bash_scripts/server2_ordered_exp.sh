#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
  printf "\n\n\n####################\n####################\n Experiment $i\n####################\n####################\n\n\n"
  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar10

  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar100

  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
  --algo_params all_algos --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
    --dataset ImageNet16-120

#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  /home/daniel_y/anaconda3/envs/knas/bin/python /home/daniel_y/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel_y/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
done

