#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
python3 $1 $2 # usage: python3 -m script.example 
