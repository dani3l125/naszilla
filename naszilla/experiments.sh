#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
  printf "\n\n\n####################\n####################\n Experiment $i\n####################\n####################\n\n\n"
  /home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 --algo_params pknas --queries 300 \
   --trials 1 --k_alg 1 --cfg /home/daniel/naszilla/naszilla/config_files/$i.yaml --output_filename pknas$i
done

