#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 30
#SBATCH -p long 
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=END

module add opencv/3.3.0
module add cuda/8.0
module add cudnn/5.1-cuda-8.0

source enet/bin/activate
python main.py -b 4 --epochs 50 --print-step 25  --name ENet --lr-decay-epochs 60 -lr 1e-3 --beta0 0.7 --validate-every 3
