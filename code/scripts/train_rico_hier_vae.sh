python ./train_rico_hier.py \
  --exp_name 'rico_hier_vae' \
  --data_path '/home/dipu/dipu_ps/codes/UIGeneration/data' \
  --train_dataset '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_gen_data/rico_mtn_50_geq2_mcpn_10/train_uxid.txt' \
  --val_dataset '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_gen_data/rico_mtn_50_geq2_mcpn_10/val_uxid.txt' \
  --epochs 100 \
  --model_version 'model_rico_hier'