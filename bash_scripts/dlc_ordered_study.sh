#!/bin/bash

TRIALS=10

#let NTHREADS=$TRIALS*2*3+4

  screen -L -Logfile cifar10_exp$i -S cifar10_exp$i -dm srun --mincpus=$NTHREADS\
   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
  cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\\
    --dataset cifar10"
#
  screen -L -Logfile cifar100_exp$i -S cifar100_exp$i -dm srun --gpus=1 --mincpus=$NTHREADS\
   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
  cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\\
    --dataset cifar100"

screen -L -Logfile imagenet_ablation -S imagenet_ablation -dm srun --gpus=1 --mincpus=8\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset ImageNet16-120 --study 1"

