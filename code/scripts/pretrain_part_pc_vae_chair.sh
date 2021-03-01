python ./pretrain_part_pc.py \
  --exp_name 'part_pc_vae_chair' \
  --category 'Chair' \
  --data_path '../data/partnetdata/chair_geo' \
  --train_dataset 'train_no_other_less_than_10_parts.txt' \
  --val_dataset 'val_no_other_less_than_10_parts.txt' \
  --epochs 200 \
  --model_version 'model_part_pc' \
  --batch_size 64 \
  --lr 1e-3 \
  --lr_decay_every 300 \
  --lr_decay_by 0.9 \
  --use_local_frame \
  --loss_weight_kldiv 0.01


