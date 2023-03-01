#!/bin/bash

TRIALS=6
NTHREADS=$TRIALS+1
#datasets=('cifar10' 'cifar100' 'ImageNet16-120')
#algos=('evolution' 'random' 'local_search')
datasets=('cifar10' 'cifar100' 'ImageNet16-120')
algos=('evolution' 'random' 'local_search')
studies=(1)
cfgs=('/dyakovlev/naszilla/naszilla/config_files/1.yaml')
exps=(1 2)
dgxs=('dgx02' 'dgx04' 'dgx06')

for index in 1 2 3; do
  for exp in ${exps[@]}; do
    for study in ${studies[@]}; do
      for algo in ${algos[@]}; do
        sbatch -w dgxs[${index}] -J "exp${exp}_${algo}_${datasets[${index}]}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG=${algo},DATA=${datasets[${index}]},STUDY=${study},CFG="/dyakovlev/naszilla/naszilla/config_files/${exp}.yaml" bash_scripts/one_experiment.sh
      done
      sbatch  -w dgx05 -J "exp${exp}_bananas_${datasets[${index}]}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG='bananas',DATA=${datasets[${index}]},STUDY=${study},CFG="/dyakovlev/naszilla/naszilla/config_files/${exp}.yaml" bash_scripts/one_experiment_bananas.sh
    done
  done
done