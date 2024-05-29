#!/bin/bash

#SBATCH -n 4
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=20g
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=a100_80gb:1
#SBATCH --gres=gpumem:20g
#SBATCH --job-name=patio
#SBATCH --output=slurm/patio.out
#SBATCH --error=slurm/patio.err

python -m train \
  --gin_configs=configs/360_dino.gin \
  --gin_bindings="Config.data_dir = 'Datasets/on-the-go/patio'" \
  --gin_bindings="Config.checkpoint_dir = 'output/patio/run_1/checkpoints'" \
  --gin_bindings="Config.patch_size = 32" \
  --gin_bindings="Config.dilate = 4" \
  --gin_bindings="Config.data_loss_type = 'on-the-go'" \
  --gin_bindings="Config.train_render_every = 5000" \
  --gin_bindings="Config.H = 1080" \
  --gin_bindings="Config.W = 1920" \
  --gin_bindings="Config.factor = 4" \
  --gin_bindings="Config.feat_rate = 2" \
