## Commands
1. To train, run the following command
```
MODEL=pointaudionav_depth && python mapnav_rl/run.py --exp-config mapnav_rl/config/train_telephone/$MODEL.yaml --model-dir data/models/mapnav_rl/telephone/$MODEL
```
2. To evaluate with all checkpoints in model directory
```
MODEL=pointaudionav_depth && py mapnav_rl/run.py --run-type eval --exp-config mapnav_rl/config/val_telephone/$MODEL.yaml --model-dir data/models/mapnav_rl/telephone/$MODEL
```
3. To test all trained models with best validation checkpoints, run the following command 
```
py mapnav_rl/test_all.py --models pointnav_depth,pointaudionav_depth --models-dir data/models/telephone --max-step 400 --event-dir eval_spl_val_telephone_average_spl
```
4. Generate demo video with audio
```
py mapnav_rl/test_all.py --models-dir data/models/mapnav_rl/telephone --eps apartment_1_17987 --models pointaudionav_depth --max-step 1000 --no-search  EVAL.SPLIT val_telephone TEST_EPISODE_COUNT 2 VISUALIZATION_OPTION [\"top_down_map\"] VIDEO_OPTION [\"disk\"] SENSORS [\"RGB_SENSOR\",\"DEPTH_SENSOR\"] ENCODE_RGB False
```