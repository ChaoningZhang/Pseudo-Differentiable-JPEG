#### NON-ROBUST DATASET ####
python3 main.py \
  'new' \
  --name 'different_qf_jpeg2_cover_dependent_adv_loss_1e-3' \
  --noise 'diff_qf_jpeg2' \
  --data-dir '/media/user/SSD1TB-1/ImageNet/' \
  --batch-size 50 \
  --cover-dependent 1 \
  --adv_loss 1e-3 \
  --save-dir 'runs_adv_loss/1e-3'
  # --save-dir 'runs_adv_loss/debug'