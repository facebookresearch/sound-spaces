# Step-by-step installation guide for SoundSpaces (1.0 and 2.0)
SoundSpaces 1.0 and SoundSpaces 2.0 are currently compatible and can be used in one environment, although each of them requires specific instructions to follow.

## Set up the conda environment
```
conda create -n ss python=3.9 cmake=3.14.0 -y
conda activate ss
```

## Install dependences (habitat-lab/habitat-sim)
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .
```
Edit habitat/tasks/rearrange/rearrange_sim.py file and remove the 36th line where FetchRobot is imported.

## Install soundspaces
```
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

## Download scene datasets
```
cd sound-spaces
mkdir data && cd data
mkdir scene_datasets && cd scene_datasets
```
Follow [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md) to download scene datasets in the folder, e.g., Replica, Matteport3D, Gibson, HM3D. Make sure to download the SceneDatasetConfig file for each dataset.

## SoundSpaces 1.0 specific instructions
SoundSpaces 1.0 provides pre-rendered RIRs and dataset configurations for AudioGoal navigation as well as semantic AudioGoal navigation tasks on two datasets (Replica and Matterport).

Follow the [dataset instructions](soundspaces/README.md) to download data needed. Note that, to use the pretrained model, you need to **temporarily** check out habitat-sim and habitat-lab both at v0.1.7 and run ```python scripts/cache_observations.py```.  Without this step, the data rendered from the newest habitat-sim is different from what used for the pretrained models.

To test if the installation is correct, run the following code
```
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/test_telephone/audiogoal_depth.yaml EVAL_CKPT_PATH_DIR data/pretrained_weights/audionav/av_nav/replica/heard.pth 
```
If the code runs fine, it should print out a success rate of 0.97 and a SPL of 0.803164.

## SoundSpaces 2.0 specific instructions
SoundSpaces 2.0 provides the ability to render IRs on the fly.

### Download material configuration file
This file assigns acoustic coefficients to different acoustic materials.
```
cd data && wget https://raw.githubusercontent.com/facebookresearch/rlr-audio-propagation/main/RLRAudioPropagationPkg/data/mp3d_material_config.json
```

### Building GLIBC
For systems with GLIBC verion below 2.29, following the instructions in this [link](https://github.com/facebookresearch/rlr-audio-propagation/issues/9#issuecomment-1317697962) to build the required GLIBC binaries.

To test if the installation is correct, run 
```
python examples/minimal_example.py
```
and it should output data/output.wav without an error.


### Current open issues
* If you run into [invalid pointer issues](https://github.com/facebookresearch/habitat-sim/issues/1747), import quaternion before habitat_sim as a workaround.
* If the audio rendering crashes due to errors in loading semantic annotations, try to set ```audio_sensor_spec.enableMaterials = False```
