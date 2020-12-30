#### NON-ROBUST DATASET ####
python3 main.py \
  'new' \
  --name 'mpeg4_cover_dependent_adv_loss_1e-1' \
  --noise 'mpeg4' \
  --data-dir '/media/user/SSD1TB-1/ImageNet/' \
  --batch-size 32 \
  --cover-dependent 1 \
  --adv_loss 1e-1 \