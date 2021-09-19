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

- Evaluate the pre-training process. This will automatically run evaluation on the `test` data split for each of the checkpoints found in `data/models/saven/data`. Use the additional flag `--prev-ckpt-ind` to instead specify a starting checkpoint index for the evaluation process, or to resume an evaluation process. 
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven --run-type eval EVAL.SPLIT test
```

- Once evaluation is complete, obtain the best checkpoint of the pre-training step and its corresponding metrics. 
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml --model-dir data/models/saven --run-type eval --eval-best EVAL.SPLIT test
```

- Train the `saven` model using the best pre-trained checkpoint of pre-training it. Please update the `pretrained_weights` path in `saven.yaml` with the best pre-trained checkpoint when finetuning.:
```
python ss_baselines/saven/run.py --exp-config ss_baselines/saven/config/semantic_audionav/saven.yaml --model-dir data/models/saven
```

## Notes 
 - Modify the parameter `NUM_UPDATES` in the configuration file according to the number of GPUs
