# Semantic Audio-Visual Navigation (SAVi) Model

## Details 
This folder provides the code of the model as well as the training/evaluation configurations used in the 
[Semantic Audio-Visual Navigation](https://arxiv.org/pdf/2012.11583.pdf) paper.
Use of this model is the similar as described in the usage section of the main README file. 
Simply replace av_nav with savi in the command.

Note that the numbers in the paper were initially reported on Habitat-Lab v0.1.5. Later versions of Habitat-Lab 
seed the random seeds a bit differently. The difference of performance should be within 1%. 
Pretrained weights are provided.

## Usage
1. Pretrain the label predictor:
```
python ss_baselines/savi/pretraining/audiogoal_trainer.py --run-type train --model-dir data/models/savi --predict-label
```
2. Train the SAVi model with the pretrained label predictor (location predictor is better trained online) with DDPPO.
Submit the slurm.sh to your slurm cluster for training. 
If cluster is not available, use the following training command to train with PPO.
SAVi is first trained with external memory size 1, which only uses the last observation.
It is then fine-tuned with the whole external memory.
```
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining.yaml --model-dir data/models/savi
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml --model-dir data/models/savi
```
3. Evaluating pretrained model
```
py ss_baselines/savi/run.py --run-type eval --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml EVAL_CKPT_PATH_DIR data/pretrained_weights/semantic_audionav/savi/best_val.pth EVAL.SPLIT test USE_SYNC_VECENV True
```

## Citation
If you use this model in your research, please cite the following paper:
```
@inproceedings{chen21semantic,
  title     =     {Semantic Audio-Visual Navigation,
  author    =     {Changan Chen and Ziad Al-Halah and Kristen Grauman},
  booktitle =     {CVPR},
  year      =     {2021}
}
```