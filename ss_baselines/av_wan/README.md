# AV-WaN Model

## Details 
This folder provides the code of the model as well as the training/evaluation configurations used in the 
[Learning to Set Waypoints for Audio-Visual Navigation](https://arxiv.org/pdf/2008.09622.pdf) paper.
Use of this model is the similar as described in the usage section of the main README file. 
Simply replace av_nav with av_wan in the command.

Note that the numbers in the paper were initially reported on Habitat-Lab v0.1.5. Later versions of Habitat-Lab 
seed the random seeds a bit differently. The difference of performance should be within 1%. 
Please re-train the model for benchmarking purposes.

## Citation
If you use this model in your research, please cite the following paper:
```
@inproceedings{chen21avwan,
  title     =     {Learning to Set Waypoints for Audio-Visual Navigation,
  author    =     {Changan Chen and Ziad Al-Halah and Kristen Grauman},
  booktitle =     {ICLR},
  year      =     {2021}
}
```