#!/bin/bash
for i in 2 3
do
  printf "\n\n\n####################\n####################\n Experiment $i\n####################\n####################\n\n\n"
  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
  --algo_params bananas --queries 300 --trials 5 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml\
    --dataset cifar100
done

