#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 16:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
python3 code/script_train_diffusion.py --server
#python3 code/script_train_classifier.py --server
