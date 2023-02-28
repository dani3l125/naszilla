#!/bin/bash

NTHREADS=7
TRIALS=5
datasets=('cifar10' 'cifar100' 'ImageNet16-120')
algos=('random' 'local_search' 'evolution')
studies=(0 1)
cfgs=('/dyakovlev/naszilla/naszilla/config_files/1.yaml')

for dataset in ${datasets[@]}; do
  for cfg in ${cfgs[@]}; do
    for study in ${studies[@]}; do
      for algo in ${algos[@]}; do
        sbatch -J "${algo}_${dataset}_cfg${cfg}_study${study}" --export=NTHREADS${NTHREADS},TRIALS${TRIALS},ALG=${algo},DATA=${dataset},STUDY=${study},CFG=${cfg} /users/feldman/dyakovlev/naszilla/naszilla/bash_scripts/one_experiment.sh
      sbatch -J "bananas_${dataset}_cfg${cfg}_study${study}" --export=NTHREADS${NTHREADS},TRIALS${TRIALS},ALG='bananas',DATA=${dataset},STUDY=${study},CFG=${cfg} /users/feldman/dyakovlev/naszilla/naszilla/bash_scripts/one_experiment_bananas.sh
      done
    done
  done
done