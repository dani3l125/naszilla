#!/bin/bash
#SBATCH -o /users/feldman/dyakovlev/naszilla/%j.out
#SBATCH -e /users/feldman/dyakovlev/naszilla/%j.err
#SBATCH -D /users/feldman/dyakovlev/naszilla
#SBATCH --time=7-00:00:00
#SBATCH --get-user-env
#SBATCH --nodes 1

echo srun --mincpus=$NTHREADS \
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params $ALG --queries 300 --trials $TRIALS --k_alg 1 --cfg $CFG\\
  --dataset $DATA --study $STUDY"

srun --mincpus=$NTHREADS \
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; printf \"\n\n\n####################\n Experiment 1 study\n####################\n\n\n\"
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params $ALG --queries 300 --trials $TRIALS --k_alg 1 --cfg $CFG\\
  --dataset $DATA --study $STUDY"