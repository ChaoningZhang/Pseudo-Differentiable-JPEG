#### NON-ROBUST DATASET ####
python3 main.py \
  'new' \
  --name 'diff_jpeg_cover_dependent' \
  --noise 'diff_jpeg' \
  --data-dir '/media/user/SSD1TB-1/ImageNet/' \
  --batch-size 32 \
  --cover-dependent 1 \
  --adv_loss 1e-2 \