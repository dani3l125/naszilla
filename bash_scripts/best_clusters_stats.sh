#!/bin/bash

for dist in lev nasbot adj path real
do
  printf "\n###################\n################### $dist ditance function statistics ###################\n###################\n"
  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/alpha.py --dist $dist

done