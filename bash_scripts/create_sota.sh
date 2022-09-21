#!/bin/bash

printf "\n\n\n####################\n####################\n Creatig sota for random search, cifar 10\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params random --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar10

printf "\n\n\n####################\n####################\n Creatig sota for local search, cifar 10\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params local_search --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar10

printf "\n\n\n####################\n####################\n Creatig sota for evolution search, cifar 10\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params evolution --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar10

printf "\n\n\n####################\n####################\n Creatig sota for bananas search, cifar 10\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params bananas --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar10

printf "\n\n\n####################\n####################\n Creatig sota for dngo search, cifar 10\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params dngo --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar10

printf "\n\n\n####################\n####################\n Creatig sota for bohamiann search, cifar 10\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params bohamiann --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar10

printf "\n\n\n####################\n####################\n Creatig sota for random search, cifar 100\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params random --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar100

printf "\n\n\n####################\n####################\n Creatig sota for local search, cifar 100\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params local_search --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar100

printf "\n\n\n####################\n####################\n Creatig sota for evolution search, cifar 100\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params evolution --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar100

printf "\n\n\n####################\n####################\n Creatig sota for bananas search, cifar 100\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params bananas --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar100

printf "\n\n\n####################\n####################\n Creatig sota for dngo search, cifar 100\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params dngo --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar100

printf "\n\n\n####################\n####################\n Creatig sota for bohamiann search, cifar 100\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params bohamiann --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset cifar100

printf "\n\n\n####################\n####################\n Creatig sota for random search, ImageNet16-120\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params random --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset ImageNet16-120

printf "\n\n\n####################\n####################\n Creatig sota for local search, ImageNet16-120\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params local_search --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset ImageNet16-120

printf "\n\n\n####################\n####################\n Creatig sota for evolution search, ImageNet16-120\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params evolution --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset ImageNet16-120

printf "\n\n\n####################\n####################\n Creatig sota for bananas search, ImageNet16-120\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params bananas --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset ImageNet16-120

printf "\n\n\n####################\n####################\n Creatig sota for dngo search, ImageNet16-120\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params dngo --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset ImageNet16-120

printf "\n\n\n####################\n####################\n Creatig sota for bohamiann search, ImageNet16-120\n####################\n####################\n\n\n"
/home/daniel/anaconda3/envs/knas/bin/python /home/daniel/naszilla/naszilla/run_experiments.py --search_space nasbench_201 \
--algo_params bohamiann --queries 300 --trials 10 --k_alg 0 --save_sota 1 \
--dataset ImageNet16-120
