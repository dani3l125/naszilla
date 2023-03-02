#!/bin/bash

TRIALS=1
NTHREADS=2
#datasets=('cifar10' 'cifar100' 'ImageNet16-120')
#algos=('evolution' 'random' 'local_search')
datasets=('cifar10' 'cifar100' 'ImageNet16-120')
algos=('bananas')
studies=(0)
exps=(3)
#dgxs=('dgx02' 'dgx04' 'dgx06')

for index in 0 1 2; do
  for exp in ${exps[@]}; do
    for study in ${studies[@]}; do
      for algo in ${algos[@]}; do
        sbatch -c ${NTHREADS} -J "exp${exp}_${algo}_${datasets[${index}]}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG=${algo},DATA=${datasets[${index}]},STUDY=${study},CFG="/dyakovlev/naszilla/naszilla/config_files/${exp}.yaml" bash_scripts/one_experiment.sh
      done
      #sbatch -c ${NTHREADS} -J "${exp}_bananas_${datasets[${index}]}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG='bananas',DATA=${datasets[${index}]},STUDY=${study},CFG="/dyakovlev/naszilla/naszilla/config_files/${exp}.yaml" bash_scripts/one_experiment_bananas.sh
    done
  done
done