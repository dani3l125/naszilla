#!/bin/bash

NTHREADS=11
TRIALS=10
#datasets=('cifar10' 'cifar100' 'ImageNet16-120')
#algos=('evolution' 'random' 'local_search')
datasets=('cifar10' 'cifar100' 'ImageNet16-120')
algos=('evolution' 'random' 'local_search')
studies=(1)
cfgs=('/dyakovlev/naszilla/naszilla/config_files/1.yaml')

for dataset in ${datasets[@]}; do
  for cfg in ${cfgs[@]}; do
    for study in ${studies[@]}; do
      for algo in ${algos[@]}; do
        sbatch -w dgx03 -J "multithread_${algo}_${dataset}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG=${algo},DATA=${dataset},STUDY=${study},CFG=${cfg} bash_scripts/one_experiment.sh
      done
      sbatch  -w dgx03 -J "multithread_bananas_${dataset}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG='bananas',DATA=${dataset},STUDY=${study},CFG=${cfg} bash_scripts/one_experiment_bananas.sh
    done
  done
done