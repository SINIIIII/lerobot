# train_static_cylinder
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=sini-21/SO101-record-grip-cylinder \
  --policy.repo_id=sini-21/groot_so101_test \
  --policy.type=groot \
  --output_dir=outputs/train/groot_so101_test \
  --job_name=groot_so101_test \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --wandb.enable=true

# train_moving_cylinder (need to remove wrong dataset)
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=sini-21/SO101-record-grip-cylinder-moving \
  --policy.repo_id=sini-21/groot_so101_test \
  --policy.type=groot \
  --output_dir=outputs/train/groot_so101_test \
  --job_name=groot_so101_test \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --wandb.enable=true

# should remove this folder
rm -rf outputs/train/groot_so101_test