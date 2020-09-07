# SoundSpaces Dataset

## Overview
The SoundSpaces dataset includes audio renderings (room impulse responses) for two datasets, metadata of each scene, episode datasets and mono sound files. 


## Download
0. Create a folder named "data" under root directory
1. Download [Replica-Dataset](https://github.com/facebookresearch/Replica-Dataset) and [Matterport3D](https://niessner.github.io/Matterport).
2. Run the command below in the root directory to cache observations for two datasets
```
python scripts/cache_observations.py
```
3. Run the commands below in the **data** directory to download partial binaural RIRs (867G), metadata (1M), datasets (77M) and sound files (13M). Note that this partial binaural RIRs only contain renderings for nodes accessible by the agent on the navigation graph. 
```
wget http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs.tar && tar xvf binaural_rirs.tar
wget http://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz && tar xvf metadata.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/datasets.tar.xz && tar xvf datasets.tar.xz
```
4. (Optional) Download the full ambisonic (3.6T for Matterport) and binaural (682G for Matterport and 81G for Replica) RIRs data by running the following script in the root directory. Remember to first back up the downloaded bianural RIR data.
```
python scripts/download_data.py --dataset mp3d --rir-type binaural_rirs
python scripts/download_data.py --dataset replica --rir-type binaural_rirs
```


## Data Folder Structure
```
    .
    ├── ...
    ├── metadata                                  # stores metadata of environments
    │   └── [dataset]
    │       └── [scene]
    │           ├── point.txt                     # coordinates of all points in mesh coordinates
    │           ├── graph.pkl                     # points are pruned to a connectivity graph
    ├── binaural_rirs                             # binaural RIRs of 2 channels
    │   └── [dataset]
    │       └── [scene]
    │           └── [angle]                       # azimuth angle of agent's heading in mesh coordinates
    │               └── [receiver]-[source].wav
    ├── datasets                                  # stores datasets of episodes of different splits
    │   └── [dataset]
    │       └── [version]
    │           └── [split]
    │               ├── [split].json.gz
    │               └── content
    │                   └── [scene].json.gz
    ├── sounds                                    # stores all 102 copyright-free sounds
    │   └── 1s_all
    │       └── [sound].wav
    ├── scene_datasets                            # scene_datasets
    │   └── [dataset]
    │       └── [scene]
    │           └── [scene].house (habitat/mesh_sementic.glb)
    └── scene_observations                        # pre-rendered scene observations
    │   └── [dataset]
    │       └── [scene].pkl                       # dictionary is in the format of {(receiver, rotation): sim_obs}
```