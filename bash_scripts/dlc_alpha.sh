#!/bin/bash

TRIALS=1

let NTHREADS=$TRIALS*4+1
for i in 1 2 3 5 6 7 8 10
do
#  screen -L -Logfile cifar10_plot$i -S cifar10_plot$i -dm srun --gpus=1 --mincpus=$NTHREADS\
#   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
#   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
#  cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
#  python /dyakovlev/naszilla/naszilla/plot.py --search_space nasbench_201 \\
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\\
#    --dataset cifar10"
#
#  screen -L -Logfile cifar100_plot$i -S cifar100_plot$i -dm srun --gpus=1 --mincpus=$NTHREADS\
#   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
#   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
#  cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment $i\n####################\n\n\n\"
#  python /dyakovlev/naszilla/naszilla/plot.py --search_space nasbench_201 \\
#  --algo_params all_algos --queries 300 --trials $TRIALS --k_alg 1 --cfg /dyakovlev/naszilla/naszilla/config_files/$i.yaml\\
#    --dataset cifar100"

  screen -L -Logfile alpha -S alpha -dm srun --gpus=0\
   --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
   /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
  cd /dyakovlev/naszilla; \\
  python /dyakovlev/naszilla/naszilla/alpha.py "

done

