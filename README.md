## SoundSpaces: Audio-Visual Navigation in 3D Environments
SoundSpaces is a realistic acoustic simulation platform for audio-visual embodied AI research. From audio-visual navigation and audio-visual exploration to echolocation, this platform expands embodied vision research to a broader scope of topics.

## Introduction Video
[<img src="https://i.imgur.com/YzlSDJj.png" width="60%">](https://youtu.be/4uiptTUyq30)

Presentation videos can be found at our [project page](http://vision.cs.utexas.edu/projects/audio_visual_navigation/).

## Motivation
Moving around in the world is naturally a multisensory experience, but today's embodied agents are deaf---restricted to solely their visual perception of the environment. We introduce audio-visual navigation for complex, acoustically and visually realistic 3D environments. We further build *SoundSpaces*: a first-of-its-kind dataset of audio renderings based on geometrical acoustic simulations for two sets of publicly available 3D environments (Matterport3D and Replica), and we instrument [Habitat](https://github.com/facebookresearch/habitat-api/blob/master/README.md) to support the new sensor, making it possible to insert arbitrary sound sources in an array of real-world scanned environments.

## Citing SoundSpaces
If you use the Habitat platform in your research, please cite the following [paper](https://arxiv.org/pdf/1912.11474.pdf):
```
@inproceedings{chen20soundspaces,
  title     =     {SoundSpaces: Audio-Visual Navigaton in 3D Environments,
  author    =     {Changan Chen and Unnat Jain and Carl Schissler and Sebastia Vicenc Amengual Gari and Ziad Al-Halah and Vamsi Krishna Ithapu and Philip Robinson and Kristen Grauman},
  booktitle =     {ECCV},
  year      =     {2020}
}
```

## Installation 
1. Install [habitat-api](https://github.com/facebookresearch/habitat-api) and [habitat-sim](https://github.com/facebookresearch/habitat-sim)
2. Install this repo into pip
```
pip install -e .
```
3. Following instructions on the [dataset](soundspaces/README.md) page to download the rendered audio data and datasets

## Usage
This repo supports benchmarking PointGoal, AudioGoal and AudioPointGoal with different sensors on two datasets. Below we show the commands for training and evaluating AudioGoal with Depth sensor on Replica, but it applies to the other two tasks, other sensors and Matterport dataset as well. 
1. Training
```
python baselines/run.py --exp-config baselines/config/replica/train_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth
```
2. Validation (evaluate each checkpoint and generate a validation curve)
```
python baselines/run.py --run-type eval --exp-config baselines/config/replica/val_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth
```
3. Test the best validation checkpoint based on validation curve
```
python baselines/run.py --run-type eval --exp-config baselines/config/replica/test_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth EVAL_CKPT_PATH_DIR data/models/replica/audiogoal_depth/data/ckpt.XXX.pth
```
4. Generate demo video with audio
```
python baselines/run.py --run-type eval --exp-config baselines/config/replica/test_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth EVAL_CKPT_PATH_DIR data/models/replica/audiogoal_depth/data/ckpt.220.pth VIDEO_OPTION [\"disk\"] TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False TASK_CONFIG.TASK.SENSORS [\"POINTGOAL_WITH_GPS_COMPASS_SENSOR\",\"SPECTROGRAM_SENSOR\",\"AUDIOGOAL_SENSOR\"] SENSORS [\"RGB_SENSOR\",\"DEPTH_SENSOR\"] EXTRA_RGB True TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE True DISPLAY_RESOLUTION 512 TEST_EPISODE_COUNT 1
```
5. Interactive demo
```
python scripts/interactive_demo.py
```

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
SoundSpaces is CC-BY-4.0 licensed, as found in the [LICENSE](LICENSE) file.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.
- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Usehttp://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Replica based task datasets, the code for generating such datasets, and trained models are under [Replica license](https://github.com/facebookresearch/Replica-Dataset/blob/master/LICENSE).