#!/bin/bash
#SBATCH --job-name=dav_nav
#SBATCH --output=data/logs/%j.out
#SBATCH --error=data/logs/%j.err
#SBATCH --gres gpu:2
#SBATCH --nodes 16
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 2
#SBATCH --mem=250GB
#SBATCH --time=4320:00
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@600
#SBATCH --partition=learnlab,learnfair

#export GLOG_minloglevel=2
#export MAGNUM_LOG=quiet
#export HABITAT_SIM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x

srun python -u -m ss_baselines.av_nav.run \
    --exp-config ss_baselines/av_nav/config/audionav/mp3d/train_telephone/audiogoal_depth_ddppo.yaml  \
    --model-dir data/models/ss2/mp3d/dav_nav CONTINUOUS True
