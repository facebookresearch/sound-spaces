# SoundSpaces-PanoIR Dataset


## Description
To facilitate future research on visual-acoustic learning, we prepare this standalone dataset that is ready
to use without any interaction with the simulation. Each example consists of panoramic images (RGB/Depth), 
an impulse response and the polar coordinate of the sound source.

## Download
Run the command below to download this recomputed dataset.
```angular2html
wget http://dl.fbaipublicfiles.com/SoundSpaces/PanoIR.zip && unzip PanoIR.zip
```

## Reproduce
Run the following commands to customize the dataset to suit your own need.
```angular2html
python PanoIR/render_panoIR.py --dataset mp3d
```