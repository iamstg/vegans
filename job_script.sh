#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 40
#SBATCH -p long 
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=END

module add opencv/3.3.0
module add cuda/8.0
module add cudnn/5.1-cuda-8.0

source enet/bin/activate
python main.py
