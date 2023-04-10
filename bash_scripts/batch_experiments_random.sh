#!/bin/bash

TRIALS=9
NTHREADS=4
datasets=('cifar10' 'cifar100' 'ImageNet16-120')
algos=('random')
#datasets=('ImageNet16-120')
studies=(0)
exps=(3 4 5)
#dgxs=('dgx02' 'dgx04' 'dgx06')

for index in 0 1 2; do
  for exp in ${exps[@]}; do
    for study in ${studies[@]}; do
      for algo in ${algos[@]}; do
        sbatch -c ${NTHREADS} -J "multi_exp${exp}_${algo}_${datasets[${index}]}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG=${algo},DATA=${datasets[${index}]},STUDY=${study},CFG="/dyakovlev/naszilla/naszilla/config_files/${exp}.yaml" bash_scripts/one_experiment.sh
      done
    done
  done
done