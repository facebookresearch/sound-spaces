import quaternion

import habitat_sim.sim
import numpy as np
from scipy.io import wavfile


backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/mp3d/UwV83HsGsw3/UwV83HsGsw3.glb"
backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
backend_cfg.load_semantic_mesh = True
backend_cfg.enable_physics = False
agent_config = habitat_sim.AgentConfiguration()
cfg = habitat_sim.Configuration(backend_cfg, [agent_config])
sim = habitat_sim.Simulator(cfg)

audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"
audio_sensor_spec.enableMaterials = True
audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
audio_sensor_spec.channelLayout.channelCount = 1
audio_sensor_spec.position = [0.0, 1.5, 0.0]
audio_sensor_spec.acousticsConfig.sampleRate = 16000
audio_sensor_spec.acousticsConfig.indirect = True
sim.add_sensor(audio_sensor_spec)

audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
audio_sensor.setAudioSourceTransform(np.array([-8.56, 1.5, 0.50]))
audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
agent = sim.get_agent(0)
new_state = sim.get_agent(0).get_state()
new_state.position = np.array([-10.57, 0, -0.25])
new_state.sensor_states = {}
agent.set_state(new_state, True)
obs = np.array(sim.get_sensor_observations()["audio_sensor"])
wavfile.write('data/output.wav', 16000, obs.T)

