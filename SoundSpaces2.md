# Documentation for SoundSpaces 2.0

SoundSpaces 2.0 builds based on the [RLRAudioPropagation](https://github.com/facebookresearch/rlr-audio-propagation) library and [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) to provide both audio-visual renderings. RLRAudioPropagation is a bi-directional ray tracer based audio simulator. Given a source location, listener location, scene geometry (3D mesh), audio materials, and some parameters (described below) it will simulate how audio waves travel from the source to arrive at the listener. The output of this simulation is an impulse response for the listener.

The C++ implementation is exposed for python users via pybind11. This document explains the various python APIs, structs, and enums. Also see the relevant [Habitat-sim python API](https://aihabitat.org/docs/habitat-sim/classes.html) doc pages.

Please refer to the [installation doc](INSTALLATION.md) doc for information on how to build SoundSpaces 2.0.

## Citation
If you find this work useful in your research, please cite the following [paper](https://arxiv.org/pdf/2206.08312.pdf):
```
@article{chen22soundspaces2,
  title     =     {SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning},
  author    =     {Changan Chen and Carl Schissler and Sanchit Garg and Philip Kobernik and Alexander Clegg and Paul Calamia and Dhruv Batra and Philip W Robinson and Kristen Grauman},
  journal   =     {NeuriPS 2022 Datasets and Benchmarks Track},
  year      =     {2022}
}
```

## List of sections
- [Acoustics configuration - RLRAudioPropagationConfiguration()](#acoustics-configuration)
- [Channel layout - RLRAudioPropagationChannelLayout()](#channel-layout)
- [Audio sensor specs - AudioSensorSpec()](#audio-sensor-specs)
- [APIs](#apis)
- [Steps to run audio simulation in python](#steps-to-run-audio-simulation-in-python)



### - Acoustics configuration
The RLRAudioPropagationConfiguration() exposes various configuration options that can be used to customize the audio simulation. This section describes the available config settings including data types and default values.

|Config name|Data Type|Default Value|Usage|
|-----------|---------|-------------|-----|
| sampleRate | int | 44100 | Sample rate for the simulated audio |
| frequencyBands | int |  4 | Number of frequency bands in the audio simulation |
| directSHOrder | int | 3 | The spherical harmonic order used for calculating direct sound spatialization for non-point sources (those with non-zero radii). It is not recommended to go above order 9. |
| indirectSHOrder | int | 1 |  The spherical harmonic order used for calculating the spatialization of indirect sound (reflections, reverb). It is not recommended to go above order 5. Increasing this value requires more rays to be traced for the results to converge properly, and uses substantially more memory (scales quadratically).  |
| threadCount | int | 1 | Number of CPU thread the simulation will use |
| updateDt | float | 0.02f | Simulation time step |
| irTime | float | 4.f | Maximum render time budget for the audio simulation |
| unitScale | float | 1.f | Unit scale for the scene. Mesh and positions are multiplied by this factor |
| globalVolume | float | 0.25f | Total initial pressure value |
| indirectRayCount | int | 5000 | Number of indirect rays that the ray tracer will use |
| indirectRayDepth | int | 200 | Maximum depth of each indirect ray cast by the ray tracer |
| sourceRayCount | int | 200 | Number of direct rays that the ray tracer will use |
| sourceRayDepth | int | 10 | Maximum depth of direct rays cast by the ray tracer |
| maxDiffractionOrder | int | 10 | The maximum number of edge diffraction events that can occur between a source and listener. This value cannot exceed 10 (compile-time limit). |
| direct | bool | true | Enable contribution from the direct rays |
| indirect | bool | true | Enable contribution from the indirect rays |
| diffraction | bool | true | Enable diffraction for the simulation |
| transmission | bool | true | Enable transmission of rays |
| meshSimplification | bool | false | Uses a series of mesh simplification operations to reduce the mesh complexity for ray tracing. Vertex welding is applied, followed by simplification using the edge collapse algorithm. |
| temporalCoherence | bool | false | Turn on/off temporal smoothing of the impulse response. This uses the impulse response from the previous simulation time step as a starting point for the next time step. This reduces the number of rays required by about a factor of 10, resulting in faster simulations, but should not be used if the motion of sources/listeners is not continuous. |
| dumpWaveFiles | bool | false | Write the wave files for different bands. Will be writted to the AudioSensorSpec's [outputDirectory](#outputDirectory) |
| enableMaterials | bool | true | Enable audio materials |
| writeIrToFile | bool | false | Write the final impulse response to a file. Will be writted to the AudioSensorSpec's [outputDirectory](#outputDirectory) |



### - Channel layout

This section describes the channel layout struct, which defines what the output will look like.

|Config name|Data Type|Default Value|Usage|
|-----------|---------|-------------|-----|
| channelType | enum | [RLRAudioPropagationChannelLayoutType](#RLRAudioPropagationChannelLayoutType).Binaural | Channel type for the simulated audio |
| channelCount | int |  2 | Number of output channels in simulated audio |



#### RLRAudioPropagationChannelLayoutType

The channel layout describes how the audio output will be experienced by the listener. Lets look at channel layout types that are currently supported.

|Enum|Usage|
|-----------|---------|
Unknown | Unknown channel layout type |
Mono | Monaural channel layout that does not have any spatial information. This layout usually has 1 channel |
Binaural | Channel layout with 2 channels that spatializes audio using an HRTF |
Ambisonics | Channel layout that encodes fully spherical spatial audio as a set of spherical harmonic basis function coefficients |


<!-- ### - Audio sensor specs

|Config name|Data Type|Default Value|Usage|
|-----------|---------|-------------|-----|
|uuid|string|""|unique identifier string to name and refer to this sensor object|
| outputDirectory(#outputDirectory) | string | "" | Output directory prefix for the simulation. Folders with outputDirectory + i should be created if you want to dump the wave files. (i = 0 indexed simulation iteration | -->
<!-- | acousticsConfig | [RLRAudioPropagationConfiguration()](#acoustics-configuration) |  Defined in the relevant section | Acoustic configuration struct that defines simulation parameters |
| channelLayout | [RLRAudioPropagationChannelLayout()](#channel-layout) |  Defined in the relevant section | Channel layout for simulated output audio | -->



### - APIs

The audio sensor is implemented in C++ and exposed to python via pybind11. Import the following to get access to the audio sensor:

|python imports|
|------|
|import habitat_sim|
|import habitat_sim.sim|

The acoustic sensor spec is part of habitat_sim

|struct/enum in habitat_sim|notes|
|------|-----|
|habitat_sim.AudioSensorSpec() | acoustic sensor spec |


To call APIs on the audio sensor, get access to the audio sensor object using the uuid.

|APIs for audio_sensor|notes|
|------|-----|
|audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]| get the audio sensor object from the habitat sim object|
| audio_sensor.setAudioSourceTransform(np.array([x, y, z])) | set the audio source location where x,y,z are floats|
|audio_sensor.reset() | Reset the simulation object to restart from a fresh context. This is the same as deleting the audio sensor and re-creating it.|

Relevant APIs on the Simulator object

|APIs for the Habitat Simulator object|notes|
|------|-----|
| sim.add_sensor(audio_sensor_spec)| Add the audio sensor. This is similar to adding any other sensors|
|obs = sim.get_sensor_observations()["audio_sensor"]|Get the impulse response. obs is a n-d array where n = channel count|



### - Steps to run audio simulation in python

Please see the [jupyter notebook](examples/soundspaces2_quick_tutorial.ipynb) for an example of how to use the python audio sensor. Follow these steps and refer to the python script for the code.

1. Create the habitat sim object and configuration.
1. Create the [AudioSensorSpec()](#audio-sensor-specs). 
1. Set the acoustic configuration ([RLRAudioPropagationConfiguration](#acoustics-configuration)) object. Set the various simulation parameters.
1. Set the channel layout ([RLRAudioPropagationChannelLayout](#channel-layout)).
1. Add the audio sensor spec to the simulation. This will create the C++ AudioSensor object.
1. Get the audio_sensor object from the list of sensors on the agent. The identifier is set under AudioSensorSpec -> uuid config.
1. Set the location of the audio source by calling audio_sensor.setAudioSourceTransform
1. Run the simulation step and get the audio sensor output sim.get_sensor_observations()["audio_sensor"]. Use the uuid defined. The output is a n-d array of floats where n is the channel count defined in RLRAudioPropagationChannelLayout


### - Some notes on the audio simulation

* The acoustic simulation is based on Monte-Carlo path tracing. The simulation is stochastic and the results will vary slightly between runs. The results will also vary based on the number of rays traced. Increasing the number of rays will increase the accuracy of the simulation, but will also increase the simulation time.
* By default, the height of the sensor is on the ground. If you want to place the sound source at certain height, you'll need to set the height on Y axis.