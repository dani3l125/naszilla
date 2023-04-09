#!/bin/bash

TRIALS=9
NTHREADS=4
datasets=('cifar100' 'ImageNet16-120')
#datasets=('ImageNet16-120')
studies=(0)
exps=(2 3 4)
#dgxs=('dgx02' 'dgx04' 'dgx06')

for index in 0 1; do
  for exp in ${exps[@]}; do
    for study in ${studies[@]}; do
      sbatch -c ${NTHREADS} -J "multi_exp${exp}_bananas_${datasets[${index}]}_study${study}" --export=NTHREADS=${NTHREADS},TRIALS=${TRIALS},ALG='bananas',DATA=${datasets[${index}]},STUDY=${study},CFG="/dyakovlev/naszilla/naszilla/config_files/${exp}.yaml" bash_scripts/one_experiment_bananas.sh
    done
  done
done