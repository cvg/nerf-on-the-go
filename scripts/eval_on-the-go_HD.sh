#!/bin/bash

#SBATCH -n 4
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=20g
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=4090:4
#SBATCH --gres=gpumem:20g
#SBATCH --job-name=yard_high
#SBATCH --output=slurm/yard_high.out
#SBATCH --error=slurm/yard_high.err

python -m eval \
  --gin_configs=configs/360_dino.gin \
  --gin_bindings="Config.data_dir = 'Datasets/on-the-go/patio'" \
  --gin_bindings="Config.checkpoint_dir = 'output/patio/run_1/checkpoints'" \
  --gin_bindings="Config.eval_train = False" \
  --gin_bindings="Config.factor = 4" \
  --gin_bindings="Config.H = 1080" \
  --gin_bindings="Config.W = 1920" \
  --gin_bindings="Config.factor = 4" \
  --gin_bindings="Config.feat_rate = 2" \


  