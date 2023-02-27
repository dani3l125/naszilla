#!/bin/bash

TRIALS=6

let NTHREADS=6

screen -L -Logfile cifar10_ablation -S cifar10_random_study -dm srun --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params random --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar10 --study 1"
#
screen -L -Logfile cifar100_ablation -S cifar100_random_study -dm srun --gpus=1 --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params random --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar100 --study 1"

screen -L -Logfile imagenet_ablation -S imagenet_random_study -dm srun --gpus=1 --mincpus=8\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params random --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset ImageNet16-120 --study 1"

screen -L -Logfile cifar10_ablation -S cifar10_evolution_study -dm srun --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params evolution --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar10 --study 1"
#
screen -L -Logfile cifar100_ablation -S cifar100_evolution_study -dm srun --gpus=1 --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params evolution --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar100 --study 1"

screen -L -Logfile imagenet_ablation -S imagenet_evolution_study -dm srun --gpus=1 --mincpus=8\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params evolution --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset ImageNet16-120 --study 1"

screen -L -Logfile cifar10_ablation -S cifar10_local_study -dm srun --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params local_search --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar10 --study 1"
#
screen -L -Logfile cifar100_ablation -S cifar100_local_study -dm srun --gpus=1 --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params local_search --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar100 --study 1"

screen -L -Logfile imagenet_ablation -S imagenet_local_study -dm srun --gpus=1 --mincpus=8\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params local_search --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset ImageNet16-120 --study 1"

screen -L -Logfile cifar10_ablation -S cifar10_bananas_study -dm srun --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params bananas --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar10 --study 1"
#
screen -L -Logfile cifar100_ablation -S cifar100_bananas_study -dm srun --gpus=1 --mincpus=$NTHREADS\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params bananas --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset cifar100 --study 1"

screen -L -Logfile imagenet_ablation -S imagenet_bananas_study -dm srun --gpus=1 --mincpus=8\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params bananas --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/1.yaml\\
  --dataset ImageNet16-120 --study 1"


