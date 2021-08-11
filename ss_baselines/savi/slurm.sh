#!/bin/bash
#SBATCH --job-name=savi
#SBATCH --output=data/logs/%j.out
#SBATCH --error=data/logs/%j.err
#SBATCH --gres gpu:2
#SBATCH --nodes 16
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 2
#SBATCH --mem=60GB
#SBATCH --time=1440:00
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@600
#SBATCH --partition=devlab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x

srun python -u -m ss_baselines.savi.run \
    --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining.yaml \
    --model-dir data/models/savi