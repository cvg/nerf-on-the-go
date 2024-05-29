#!/bin/bash

#SBATCH -n 4
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=20g
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=4090:4
#SBATCH --gres=gpumem:20g
#SBATCH --job-name=patio_high
#SBATCH --output=slurm/patio_high.out
#SBATCH --error=slurm/patio_high.err

python -m render \
  --gin_configs=configs/360_dino.gin \
  --gin_bindings="Config.data_dir = 'Datasets/on-the-go/patio_high'" \
  --gin_bindings="Config.checkpoint_dir = 'output/patio_high/run_1/checkpoints'" \
  --gin_bindings="Config.render_dir = 'output/patio_high/run_1/checkpoints'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 160" \
  --gin_bindings="Config.render_video_fps = 160" \
  --gin_bindings="Config.H = 1080" \
  --gin_bindings="Config.W = 1920" \
  --gin_bindings="Config.factor = 4" \
  --gin_bindings="Config.feat_rate = 2" \
  