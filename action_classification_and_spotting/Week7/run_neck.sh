#!/bin/bash
#SBATCH -n 8 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/priubrogent/mcv-c6-2026-team4/action_classification_and_spotting/Week6/ # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 35048 # 2GB solicitados.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:2

python /hhome/priubrogent/mcv-c6-2026-team4/action_classification_and_spotting/Week7/run_experiments.py --gpus 0 1 --models phaseA_unet_h440_l2 phaseA_unet_h256_l2 phaseA_unet_h128_l2 phaseA_unet_h64_l2 phaseA_unet_h32_l2
