#!/bin/bash

screen -L -Logfile alpha -S alpha -dm srun --gpus=0\
 --container-image=/users/feldman/dyakovlev/knas.sqsh --container-mounts=/users/feldman/dyakovlev/:/dyakovlev \
 /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/naszilla\";export PYTHONPATH=\"${PYTHONPATH}:/dyakovlev/nasbench\";
cd /dyakovlev/naszilla; \\
python /dyakovlev/naszilla/naszilla/alpha.py "


