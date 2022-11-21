#!/bin/bash

TRIALS=4

let NTHREADS=$TRIALS*4+1
for i in 1 2 3 4 5 6 7 8
do
#  screen -L -Logfile cifar10_exp$i -S cifar10_exp$i -dm srun --gpus=1 --mincpus=$NTHREADS\
#   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
#   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
#  cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\\
#    --dataset cifar10"
#
#  screen -L -Logfile cifar100_exp$i -S cifar100_exp$i -dm srun --gpus=1 --mincpus=$NTHREADS\
#   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
#   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
#  cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\\
#    --dataset cifar100"

  screen -L -Logfile imagenet_exp$i -S imagenet_exp$i -dm srun --gpus=1 --mincpus=$NTHREADS\
   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
  cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\\
    --dataset ImageNet16-120"

#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params random --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params evolution --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar10
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset cifar100
#
#  python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
#  --algo_params local_search --queries 300 --trials 5 --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\
#    --dataset ImageNet16-120
done

