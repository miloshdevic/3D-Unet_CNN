#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --job-name=breast_Unet_test1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=milosh.devic@gmail.com


module load python/3.8.10
source ~/ENV/bin/activate
python ~/projects/def-senger/mdevic31/model_3D-Unet.py


