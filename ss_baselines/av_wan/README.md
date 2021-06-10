# Audio-Visual Waypoints (AV-WaN) Model

## Details 
This folder provides the code of the model as well as the training/evaluation configurations used in the 
[Learning to Set Waypoints for Audio-Visual Navigation](https://arxiv.org/pdf/2008.09622.pdf) paper.
Use of this model is the similar as described in the usage section of the main README file. 
Simply replace av_nav with av_wan in the command.

Note that the numbers in the paper were initially reported on Habitat-Lab v0.1.5. Later versions of Habitat-Lab 
seed the random seeds a bit differently. The difference of performance should be within 1%. 
Pretrained weights are provided.


## Evaluating pretrained model
```
py ss_baselines/av_wan/run.py --run-type eval --exp-config ss_baselines/av_wan/config/audionav/replica/test_with_am.yaml EVAL_CKPT_PATH_DIR data/pretrained_weights/audionav/av_wan/replica/heard.pth
py ss_baselines/av_wan/run.py --run-type eval --exp-config ss_baselines/av_wan/config/audionav/replica/test_with_am.yaml EVAL_CKPT_PATH_DIR data/pretrained_weights/audionav/av_wan/replica/unheard.pth EVAL.SPLIT test_multiple_unheard 
```


## Citation
If you use this model in your research, please cite the following paper:
```
@inproceedings{chen21avwan,
  title     =     {Learning to Set Waypoints for Audio-Visual Navigation,
  author    =     {Changan Chen, Sagnik Majumder, Ziad Al-Halah, Ruohan Gao, Santhosh K. Ramakrishnan, Kristen Grauman},
  booktitle =     {ICLR},
  year      =     {2021}
}
```