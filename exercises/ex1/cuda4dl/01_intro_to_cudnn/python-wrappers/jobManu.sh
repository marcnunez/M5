#!/bin/bash
#SBATCH -n 2 # Number of cores
#SBATCH --mem 20000 # 2GB solicitados.
#SBATCH -D /home/grupo07/M5/ex1/cuda4dl/01_intro_to_cudnn/python-wrappers/ # working directory
#SBATCH -p mhigh,mlow # or mlow Partition to submit to
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o home/grupo07/logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e home/grupo07/logs/%x_%u_%j.err # File to which STDERR will be written
#SBATCH --job-name jobManu

python3 convolution.py
