#!/bin/sh
#BSUB -q gpua100
#BSUB -J rmlsJob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 20:00
#BSUB -R "rusage[mem=25GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
python3 code/script_train_diffusion.py --server
#python3 code/script_train_predictor.py --server
