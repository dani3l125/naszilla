#!/bin/bash
#SBATCH -o %j
#SBATCH -e %j
#SBATCH -D /users/feldman/dyakovlev/knas
#SBATCH --time=7-00:00:00
#SBATCH --get-user-env
#SBATCH --nodes 1

NTHREADS=$1
TRIALS=$2
ALG=$3
DATA=$4
STUDY=$5
CFG=$6

srun --mincpus=$NTHREADS \
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla;
python /dyakovlev/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \\
--algo_params $ALG --queries 300 --trials $TRIALS --k_alg 1 --cfg $CFG\\
  --dataset $DATA --study $STUDY"