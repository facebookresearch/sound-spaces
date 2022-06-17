#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_PORT=8640

set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 2 \
    ss_baselines/av_nav/run.py \
    --exp-config ss_baselines/av_nav/config/audionav/mp3d/train_telephone/audiogoal_depth_ddppo.yaml \
    --model-dir data/models/ss2/mp3d/depth_ddppo \
    CONTINUOUS True