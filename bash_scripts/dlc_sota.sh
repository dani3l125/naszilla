#!/bin/bash

TRIALS=10

screen -L -Logfile cifar10_sota -S cifar10_sota -dm srun\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla;
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params all_algos --queries 300 --trials $TRIALS --k_alg 0 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar10"

screen -L -Logfile cifar100_sota -S cifar100_sota -dm srun\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla;
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params all_algos --queries 300 --trials $TRIALS --k_alg 0 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar100"

screen -L -Logfile imagenet_sota -S imagenet_sota -dm srun\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla;
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params all_algos --queries 300 --trials $TRIALS --k_alg 0 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset ImageNet16-120"
