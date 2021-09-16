# Semantic Audio-Visual Embodied Navigation (saven) Model

## Usage

- Pre-training the vision model:
```
python ss_baselines/saven/pretraining/vision_model_trainer.py --run-type train
```

- Pre-training the audio model:
```
python ss_baselines/saven/pretraining/audio_model_trainer.py --run-type train
```

- Pre-train the `saven` model (using the pre-trained vision and audio model). `Saven` is first trained with the external memory size of 1, which only uses the last observation.
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven
```

- Train the `saven` model using the best pre-trained checkpoint of pre-training it. Please update the `pretrained_weights` path in `saven.yaml` with the best pre-trained checkpoint when finetuning.:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven.yaml --model-dir data/models/saven
```
