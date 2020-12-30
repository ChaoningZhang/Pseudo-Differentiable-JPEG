#### NON-ROBUST DATASET ####
python3 main.py \
  'new' \
  --name 'jpeg2_cover_dependent_adv_loss_100' \
  --noise 'jpeg2' \
  --data-dir '/media/user/SSD1TB-1/ImageNet/' \
  --batch-size 32 \
  --cover-dependent 1 \
  --adv_loss 1e-2 \
  --save-dir 'runs_adv_loss/1e-2'
  # --save-dir 'runs_adv_loss/debug'